import os
from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path

# Get /path/to/my_project/app
current_dir = Path(__file__).resolve()

# Get /path/to/my_project/faiss_index
folder_path = current_dir.parent / "docs"


def load_documents_from_folder(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                docs.append(file.read())
    print(docs)
    return docs

model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast, good
documents = load_documents_from_folder(folder_path)
embeddings = model.encode(documents)

client = chromadb.Client()

collection = client.create_collection(name="my_docs")

collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

def get_relevant_docs(query_text):
    #query_text = "Wer ist besonders von PTBS in Konflikten betroffen?"
    # query_text = "Was sind SDGs?"
    print(query_text)
    query_embedding = model.encode([query_text])
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=1
    )
    print("Relevant documents:")
    for doc in results["documents"][0]:
        print(doc)

    return results


