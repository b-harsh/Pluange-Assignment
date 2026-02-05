import streamlit as st
import faiss
import pickle
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq

st.set_page_config(page_title="Pluang AI Copilot", layout="wide")


st.markdown("""
<style>

/* Make entire app font bigger */
html, body, [class*="css"] {
    font-size: 18px !important;
}

/* Chat input box styling */
div[data-testid="stChatInput"] textarea {
    font-size: 18px !important;
    border: 2px solid #4CAF50 !important;
    border-radius: 12px !important;
    padding: 12px !important;
    background-color: #f9fff9 !important;
}

/* Highlight when focused */
div[data-testid="stChatInput"] textarea:focus {
    border-color: #2E7D32 !important;
    box-shadow: 0 0 6px rgba(76, 175, 80, 0.6);
}

/* Chat bubbles font */
div[data-testid="stChatMessage"] {
    font-size: 17px !important;
    line-height: 1.6 !important;
}

/* Make titles bigger */
h1 {
    font-size: 36px !important;
}

h2, h3 {
    font-size: 26px !important;
}

</style>
""", unsafe_allow_html=True)

st.title(" Pluang Knowledge Base Copilot")
st.write("Ask questions about internal documents")



if "messages" not in st.session_state:
    st.session_state.messages = []


if st.button(" Clear Chat"):
    st.session_state.messages = []
    st.rerun()


load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedder = load_embedder()

@st.cache_resource
def load_index():

    base_dir = os.path.dirname(__file__)
    backend_dir = os.path.join(os.getcwd(), "backend")
    vector_dir = os.path.join(backend_dir, "vector_db")

    index_path = os.path.join(vector_dir, "index.faiss")
    meta_path = os.path.join(vector_dir, "meta.pkl")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Missing FAISS index: {index_path}")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing metadata: {meta_path}")

    index = faiss.read_index(index_path)

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    return index, meta


index, meta = load_index()

docs_list = meta["docs"]
sources_list = meta["sources"]



def search_docs(query, k=4):

    q_embedding = embedder.encode([query])

    D, I = index.search(np.array(q_embedding), k)

    results = []

    for idx in I[0]:

        results.append({
            "text": docs_list[idx],
            "source": sources_list[idx]
        })

    return results



def generate_answer(query, context):

    prompt = f"""
You are an internal company knowledge assistant.

Use ONLY the information below.

Context:
{context}

Question:
{query}

Answer clearly and cite sources.
"""

    response = client.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],

        temperature=0.2,
    )

    return response.choices[0].message.content



for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask your question...")

if query:

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)


    with st.spinner(" Searching documents..."):

        docs = search_docs(query)

        context = ""
        sources = []

        for i, d in enumerate(docs, 1):

            context += f"[Source {i}] {d['text']}\n\n"
            sources.append(d["source"])


    with st.spinner(" Generating answer..."):

        answer = generate_answer(query, context)


    sources_text = "\n".join(f"- {s}" for s in set(sources))

    final_answer = f"""
{answer}

---

###  Sources
{sources_text}
"""


    st.session_state.messages.append({
        "role": "assistant",
        "content": final_answer
    })


    with st.chat_message("assistant"):
        st.markdown(final_answer)