import streamlit as st
from pdf_utils import extract_text_from_pdf
from rag_bot import answer_query, chat_history

st.set_page_config(page_title="Smart FAQ Bot", layout="wide")
st.title("📘 Smart FAQ Bot (PDF + Cohere + Chat)")

uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        text = extract_text_from_pdf(uploaded_file)
        chunks = text.split("\n\n")  # basic chunking

    question = st.text_input("Ask a question about the document:")
    if question:
        with st.spinner("Thinking..."):
            answer, sources = answer_query(question, chunks)
            st.markdown("### 🤖 Answer")
            st.write(answer)

            with st.expander("📚 Show source context"):
                for idx, src in enumerate(sources, 1):
                    st.markdown(f"**Context {idx}:**\n{src}")

    st.markdown("---")
    st.markdown("### 🧠 Chat History")
    for chat in reversed(chat_history):
        st.markdown(f"**Q:** {chat['question']}")
        st.markdown(f"**A:** {chat['answer']}")
