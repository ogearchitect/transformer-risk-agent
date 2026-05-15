from __future__ import annotations

import streamlit as st

from src.agents.orchestrator import handle_query

EXAMPLE_PROMPTS = [
    "Show me the top 5 highest-risk transformers and why",
    "What's the failure probability for T-0042 at 5 years?",
    "Recommend a 5-year replacement plan with a $15M budget",
    "How many 230 kV spares should we hold?",
    "Run a what-if: defer T-0104 overhaul by 2 years",
    "Explain the DGA situation on T-0217 in plain English",
]


def _render_assistant_message(resp) -> None:
    with st.chat_message("assistant"):
        st.markdown(resp.answer)
        st.markdown("**Plan**")
        st.write(resp.plan)
        for call in resp.tool_calls:
            with st.expander(f"🔧 used tool: `{call['tool']}`"):
                st.json({"args": call["args"], "result": call["result"]})


def render_chat_panel() -> None:
    with st.sidebar:
        st.markdown("### 🤖 Agent Chat")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for prompt in EXAMPLE_PROMPTS:
            if st.button(prompt, key=f"chip-{prompt}"):
                st.session_state["_queued_prompt"] = prompt

        for m in st.session_state.messages[-8:]:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        queued = st.session_state.pop("_queued_prompt", None)
        user_msg = queued or st.chat_input("Ask about risk, strategy, or what-if...")
        if user_msg:
            st.session_state.messages.append({"role": "user", "content": user_msg})
            with st.chat_message("user"):
                st.markdown(user_msg)
            resp = handle_query(user_msg)
            st.session_state.messages.append({"role": "assistant", "content": resp.answer})
            _render_assistant_message(resp)
