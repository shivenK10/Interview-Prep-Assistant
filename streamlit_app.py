import streamlit as st
from generation_pipeline import generate_answer
import time

st.set_page_config(
    page_title="Interview Prep Assistant",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed"
)

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Context chunks", min_value=1, max_value=10, value=5)
    show_sources = st.checkbox("Show sources", value=True)
    show_chunks = st.checkbox("Show retrieved context", value=False)

    st.divider()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.divider()
    st.caption("Powered by Llama 3.2 3B & Qdrant")

st.title("Interview Prep Assistant")
st.caption("Chat with your interview preparation materials")

for entry in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(entry["question"])

    with st.chat_message("assistant"):
        st.markdown(entry["answer"])

        if show_sources and entry.get("sources"):
            with st.expander("View sources"):
                for source in entry["sources"]:
                    st.markdown(f"- {source}")

        if show_chunks and entry.get("chunks"):
            with st.expander("View retrieved context"):
                for i, chunk in enumerate(entry["chunks"], 1):
                    st.markdown(
                        f"**[{i}] {chunk['source']} - Page {chunk['page']}** "
                        f"(Score: {chunk['score']:.3f})"
                    )
                    st.text(chunk["text"][:500] + "...")
                    st.divider()

if not st.session_state.history:
    with st.chat_message("assistant"):
        st.markdown("Hi! Ask me anything about your interview prep materials 😊")
        st.markdown("**Example questions:**")
        st.markdown("- What is the OSI model?")
        st.markdown("- Explain TCP/IP layers")
        st.markdown("- What does a Data Scientist do?")
        st.markdown("- What skills are needed for a Business Analyst?")

prompt = st.chat_input("Type your question here...")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start_time = time.time()
                result = generate_answer(prompt, top_k=top_k)
                elapsed_time = time.time() - start_time

                st.markdown(result["answer"])

                if show_sources and result.get("sources"):
                    with st.expander("View sources"):
                        for source in result["sources"]:
                            st.markdown(f"- {source}")

                if show_chunks and result.get("retrieved_chunks"):
                    with st.expander("View retrieved context"):
                        for i, chunk in enumerate(result["retrieved_chunks"], 1):
                            st.markdown(
                                f"**[{i}] {chunk['source']} - Page {chunk['page']}** "
                                f"(Score: {chunk['score']:.3f})"
                            )
                            st.text(chunk["text"][:500] + "...")
                            st.divider()

        st.session_state.history.append({
            "question": prompt,
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "chunks": result.get("retrieved_chunks", []),
            "time": elapsed_time,
        })

    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Error: {str(e)}")
