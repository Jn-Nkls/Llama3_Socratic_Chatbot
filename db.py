import os
from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path

current_dir = Path(__file__).resolve()

folder_path = current_dir.parent / "docs"

print('DB')
def load_documents_from_folder(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                docs.append(file.read())
    #print(docs)
    return docs

model = SentenceTransformer('all-MiniLM-L6-v2')
documents = load_documents_from_folder(folder_path)
embeddings = model.encode(documents)
#print(documents)

client = chromadb.Client()

collection = client.create_collection(name="my_docs")

collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))]
)
#print(collection.get())
query_text = "Was sind SDGs?"

def get_relevant_docs(query_text):
    #query_text = "Wer ist besonders von PTBS in Konflikten betroffen?"
    #query_text = "Was sind SDGs?"
    query_embedding = model.encode([query_text])
    #print(query_embedding)
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=1
    )
    #print("Relevant documents:")
    for doc in results["documents"][0]:
        print(doc)

    return results
get_relevant_docs(query_text)

