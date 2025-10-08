# Git Klonen
`git clone https://github.com/Jn-Nkls/Llama3_Socratic_Chatbot.git`

# Windows
1. `download pytorch (cuda)`
2. Ollama runterladen: https://ollama.com/download/windows
3. `pip install ollama`
4. Ollama starten und danach Modell herunterladen: `ollama pull llama:8b`
5. `pip install streamlit`
6. `pip install langchain`
7. (`pip install plotly`)
8. (`pip install pymupdf`)
9. (`pip install --upgrade huggingface_hub sentence-transformers transformers accelerate`)
10. `hf download cross-encoder/ms-marco-MiniLM-L-6-v2 --local-dir "models/cross-encoder-ms-marco-MiniLM-L-6-v2"`
11. `hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir "models/all-MiniLM-L6-v2"`


5. `streamlit run app.py`


# Linux
1. `download pytorch (cuda)`
2. Ollama runterladen: `curl -fsSL https://ollama.com/install.sh | sh`
3. Um Ollama zu starten: `ollama serve`
4. Llama3 herunterladen: `ollama pull llama:8b`
5. `sudo apt install python3-pip`
6. `pip install langchain`
7. `pip install streamlit`
8. `pip install ollama`
9. (`pip install pymupdf`)
10. (`pip install --upgrade huggingface_hub sentence-transformers transformers accelerate`)
11. `hf download cross-encoder/ms-marco-MiniLM-L-6-v2 --local-dir "models/cross-encoder-ms-marco-MiniLM-L-6-v2"`
12. `hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir "models/all-MiniLM-L6-v2"`
13. `streamlit run app.py`

