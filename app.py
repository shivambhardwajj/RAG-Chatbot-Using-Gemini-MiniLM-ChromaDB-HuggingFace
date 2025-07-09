
import streamlit as st
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from tempfile import NamedTemporaryFile
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import shutil

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
CHROMA_DIR = "chroma_store"

st.set_page_config(page_title="Gemini RAG Chatbot", layout="wide")
st.title("🤖 Gemini 2.5 Flash RAG Chatbot")
st.caption("Built with Chroma, MiniLM, and Gemini 2.5 Flash")

with st.sidebar:
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
    build_kb = st.button("📚 Build Knowledge Base")

if build_kb and uploaded_files:
    with st.spinner("Reading and embedding documents..."):
        all_chunks = []
        for file in uploaded_files:
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getvalue())
                loader = PyMuPDFLoader(tmp.name)
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_documents(docs)
                all_chunks.extend(chunks)

        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        os.makedirs(CHROMA_DIR, exist_ok=True)

        try:
            vectorstore = Chroma.from_texts([chunk.page_content for chunk in all_chunks],
                                            embedding=embedder,
                                            persist_directory=CHROMA_DIR)
            vectorstore.persist()
            st.session_state.vectorstore = vectorstore
            st.success("✅ Knowledge base built!")
        except Exception as e:
            st.error(f"Chroma initialization failed: {e}")

if "vectorstore" in st.session_state:
    st.markdown("---")
    st.header("💬 Ask a Question")
    user_input = st.text_input("Your Question:")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.button("🔍 Ask") and user_input:
        with st.spinner("Thinking with Gemini 2.5 Flash..."):
            retriever = st.session_state.vectorstore.as_retriever()
            relevant_docs = retriever.get_relevant_documents(user_input)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            prompt = f"""
You are a helpful assistant. Use the provided context to answer the question.

Context:
{context}

Question: {user_input}

Answer:""".strip()

            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            answer = response.text
            st.session_state.chat_history.append((user_input, answer))

    if st.session_state.chat_history:
        st.subheader("📜 Chat History")
        for q, a in reversed(st.session_state.chat_history):
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Gemini:** {a}")

st.markdown("---")
st.markdown("Built by Shivam Bhardwaj · Powered by Google Gemini · Chroma · HuggingFace")
