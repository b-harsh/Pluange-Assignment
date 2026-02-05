import os
import pickle

import faiss
from sentence_transformers import SentenceTransformer

from config import DOCS_PATH, VECTOR_DB_PATH


def load_documents():
    docs = []
    sources = []

    for file in os.listdir(DOCS_PATH):
        path = os.path.join(DOCS_PATH, file)

        if file.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

                chunks = text.split("\n")

                for chunk in chunks:
                    if chunk.strip():
                        docs.append(chunk)
                        sources.append(file)

    return docs, sources


def main():
    print("Loading documents...")

    docs, sources = load_documents()

    print(f"Loaded {len(docs)} chunks")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Creating embeddings...")

    embeddings = model.encode(docs)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs(VECTOR_DB_PATH, exist_ok=True)

    faiss.write_index(index, f"{VECTOR_DB_PATH}/index.faiss")

    with open(f"{VECTOR_DB_PATH}/meta.pkl", "wb") as f:
        pickle.dump(
            {
                "docs": docs,
                "sources": sources,
            },
            f,
        )

    print("Vector store created successfully!")


if __name__ == "__main__":
    main()