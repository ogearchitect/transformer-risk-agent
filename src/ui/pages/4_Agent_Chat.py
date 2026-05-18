from __future__ import annotations

import os

import streamlit as st

from src.agents.orchestrator import handle_query
from src.ui.chat_panel import EXAMPLE_PROMPTS, _render_assistant_message
from src.ui.components import branded_header, footer_badge, inject_global_styles

st.set_page_config(page_title="Agent Chat", page_icon="⚡", layout="wide")

USER_AVATAR = "🧑"
SESSION_KEY = "agent_chat_messages"


def _render_user_message(content: str) -> None:
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(content)


def _process_prompt(prompt: str) -> None:
    messages = st.session_state.setdefault(SESSION_KEY, [])
    messages.append({"role": "user", "content": prompt})
    with st.spinner("Thinking…"):
        try:
            resp = handle_query(prompt)
        except Exception as exc:  # noqa: BLE001
            messages.append(
                {
                    "role": "assistant",
                    "content": f"⚠️ Agent error: `{type(exc).__name__}: {exc}`",
                    "plan": [],
                    "tool_calls": [],
                }
            )
            st.rerun()
            return
    messages.append(
        {
            "role": "assistant",
            "content": resp.answer,
            "plan": list(resp.plan or []),
            "tool_calls": list(resp.tool_calls or []),
        }
    )
    st.rerun()


def _latest_assistant_message() -> dict | None:
    for message in reversed(st.session_state.get(SESSION_KEY, [])):
        if message.get("role") == "assistant":
            return message
    return None


def _render_session_panel() -> None:
    messages = st.session_state.get(SESSION_KEY, [])
    user_count = sum(1 for m in messages if m.get("role") == "user")
    assistant_count = sum(1 for m in messages if m.get("role") == "assistant")
    latest = _latest_assistant_message() or {}
    tool_calls = list(latest.get("tool_calls") or [])
    plan = list(latest.get("plan") or [])

    with st.container(border=True):
        st.markdown("#### Session")
        c1, c2 = st.columns(2)
        c1.metric("Turns", user_count)
        c2.metric("Tool calls", len(tool_calls))

    with st.container(border=True):
        st.markdown("#### Latest plan")
        if plan:
            st.markdown("\n".join(f"{idx}. {step}" for idx, step in enumerate(plan, start=1)))
        else:
            st.caption("No plan yet — ask the agent something.")

    with st.container(border=True):
        st.markdown("#### Latest tool calls")
        if not tool_calls:
            st.caption("No tools have been invoked yet.")
        else:
            for idx, call in enumerate(tool_calls, start=1):
                st.markdown(f"**{idx}. `{call.get('tool', 'tool')}`**")
                st.caption("Args")
                st.code(str(call.get("args", {})), language="json")
                if idx < len(tool_calls):
                    st.divider()

    if st.button("🗑 Clear conversation", use_container_width=True, key="agent_chat_clear"):
        st.session_state[SESSION_KEY] = []
        st.session_state.pop("_agent_chat_queued_prompt", None)
        st.rerun()


def main() -> None:
    inject_global_styles()
    branded_header(
        "Agent Chat",
        "Talk to the orchestrator — it routes your question to the right transformer-risk tools",
    )

    st.session_state.setdefault(SESSION_KEY, [])

    st.markdown("#### Try a demo prompt")
    chip_cols = st.columns(len(EXAMPLE_PROMPTS))
    for col, prompt in zip(chip_cols, EXAMPLE_PROMPTS):
        with col:
            if st.button(prompt, key=f"agent-chip-{prompt}", use_container_width=True):
                st.session_state["_agent_chat_queued_prompt"] = prompt

    chat_col, info_col = st.columns([68, 32], gap="large")

    with chat_col:
        with st.container(border=True):
            st.markdown("#### Conversation")
            messages = st.session_state.get(SESSION_KEY, [])
            if not messages:
                st.info("Ask anything about your fleet — top risks, replacement plans, what-if scenarios, spares, or a single asset.")
            for message in messages:
                if message["role"] == "assistant":
                    _render_assistant_message(message)
                else:
                    _render_user_message(message.get("content", ""))

        queued = st.session_state.pop("_agent_chat_queued_prompt", None)
        typed = st.chat_input("Ask the orchestrator…", key="agent_chat_input")
        prompt = queued or typed
        if prompt:
            _process_prompt(prompt)

    with info_col:
        _render_session_panel()

    model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", os.getenv("OPENAI_MODEL", "MockLLM"))
    footer_badge("Mock LLM" if model_name == "MockLLM" else f"Azure AI Foundry — {model_name}")


if __name__ == "__main__":
    main()
