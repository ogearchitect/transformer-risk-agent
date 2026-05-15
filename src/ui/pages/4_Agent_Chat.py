from __future__ import annotations

import streamlit as st

from src.agents.orchestrator import handle_query
from src.ui.components import branded_header

st.set_page_config(page_title="Agent Chat", page_icon="⚡", layout="wide")


def main() -> None:
    branded_header()
    st.markdown("### Full Agent Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Ask the orchestrator...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        resp = handle_query(prompt)
        with st.chat_message("assistant"):
            st.markdown(resp.answer)
            st.markdown("**Plan**")
            st.write(resp.plan)
            for call in resp.tool_calls:
                with st.expander(f"🔧 used tool: `{call['tool']}`"):
                    st.json(call)

        st.session_state.messages.append({"role": "assistant", "content": resp.answer})


if __name__ == "__main__":
    main()
