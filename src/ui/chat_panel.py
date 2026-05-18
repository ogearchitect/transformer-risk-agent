from __future__ import annotations

import json

import streamlit as st

from src.agents.orchestrator import handle_query

USER_AVATAR = "🧑"
ASSISTANT_AVATAR = "⚡"
RESULT_PREVIEW_LIMIT = 500

EXAMPLE_PROMPTS = [
    "Show me the top 5 highest-risk transformers and why",
    "Recommend a 5-year replacement plan with a $15M budget",
    "Run a what-if: defer T-0104 overhaul by 2 years",
    "How many 230 kV spares should we hold?",
    "What's the failure probability for T-0042 at 5 years?",
]


def _stringify(value: object) -> str:
    if isinstance(value, str):
        return value

    try:
        return json.dumps(value, indent=2, default=str, ensure_ascii=False)
    except TypeError:
        return str(value)


def _truncate_preview(value: object, limit: int = RESULT_PREVIEW_LIMIT) -> str:
    text = _stringify(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}…"


def _get_assistant_parts(resp) -> tuple[str, list[str], list[dict]]:
    if isinstance(resp, dict):
        return (
            str(resp.get("content", "")),
            list(resp.get("plan", []) or []),
            list(resp.get("tool_calls", []) or []),
        )

    return (
        str(getattr(resp, "answer", "")),
        list(getattr(resp, "plan", []) or []),
        list(getattr(resp, "tool_calls", []) or []),
    )


def _render_plan(plan: list[str]) -> None:
    if not plan:
        return

    st.markdown("**Plan**")
    st.markdown("\n".join(f"{idx}. {step}" for idx, step in enumerate(plan, start=1)))


def _render_tool_calls(tool_calls: list[dict]) -> None:
    if not tool_calls:
        return

    with st.expander(f"🔧 {len(tool_calls)} tool calls", expanded=False):
        for idx, call in enumerate(tool_calls, start=1):
            st.markdown(f"**{idx}. `{call.get('tool', 'tool')}`**")
            st.caption("Args")
            st.code(_stringify(call.get("args", {})), language="json")
            st.caption("Result preview")
            st.code(_truncate_preview(call.get("result")), language="text")
            if idx < len(tool_calls):
                st.divider()


def _render_assistant_message(resp) -> None:
    answer, plan, tool_calls = _get_assistant_parts(resp)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        st.markdown(answer)
        _render_plan(plan)
        _render_tool_calls(tool_calls)


def render_chat_panel() -> None:
    with st.sidebar:
        st.markdown("### 🤖 Agent Chat")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        st.caption("Try a demo prompt")
        prompt_cols = st.columns(len(EXAMPLE_PROMPTS))
        for col, prompt in zip(prompt_cols, EXAMPLE_PROMPTS):
            with col:
                if st.button(prompt, key=f"chip-{prompt}", use_container_width=True):
                    st.session_state["_queued_prompt"] = prompt

        for message in st.session_state.messages[-8:]:
            if message["role"] == "assistant":
                _render_assistant_message(message)
            else:
                with st.chat_message("user", avatar=USER_AVATAR):
                    st.markdown(message["content"])

        queued = st.session_state.pop("_queued_prompt", None)
        user_msg = queued or st.chat_input("Ask about risk, strategy, or what-if...")
        if user_msg:
            st.session_state.messages.append({"role": "user", "content": user_msg})
            with st.chat_message("user", avatar=USER_AVATAR):
                st.markdown(user_msg)
            with st.spinner("Thinking..."):
                resp = handle_query(user_msg)
            assistant_message = {
                "role": "assistant",
                "content": resp.answer,
                "plan": resp.plan,
                "tool_calls": resp.tool_calls,
            }
            st.session_state.messages.append(assistant_message)
            _render_assistant_message(assistant_message)

        if st.button("🗑 Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pop("_queued_prompt", None)
            st.rerun()
