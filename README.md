# BNE-Chatbot

Dieser Chatbot wurde vom URZ der Universität Leipzig im Kontext des Projektes "Digitale Nachhaltigkeit in der Lehre" 
(DiNaLe) entwickelt. Ziel war es, einen Chatbot bereitzustellen, welcher dabei unterstützt, Nachhaltigkeit didaktisch 
wirksam zu vermitteln.

Dafür wird didaktisch auf das Prinzip des sokratischen Dialogs gesetzt. Dies bedeutet, dass der Chatbot keine 
Antworten mit fachlichen Informationen gibt, sondern anregende/kritische Fragen stellt, die zu einer intensiveren 
Beschäftigung mit der Thematik anregen soll. Dadurch soll erreicht werden, dass die Nutzenden sich Schritt für Schritt 
die Antworten auf ihre Fragen selbst erarbeiten und bereits vorhandenes Wissen kritisch reflektiert wird.

Implementiert wurde der Chatbot im Moodle-Kurs “Handlungskompetenz für nachhaltige Entwicklung”. In diesem Kurs wird der Chatbot
als Brainstorming-Tool für die Vorbereitung auf die kommende Vorlesung und die Bearbeitung von Aufgaben für diese 
eingesetzt. Dafür benötigt der Chatbot für die Vorlesung relevante Daten, weshalb die RAG-Architektur verwendet wurde. 
Der Chatbot verwendet dementsprechend kein eigens trainiertes KI-Modell, sondern greift auf eine lokale Datenbank mit 
vorlesungsrelevanter Literatur zurück.

---

# Technische Spezifikationen

Der Chatbot basiert auf dem lokalen KI-Modell **llama3:8b** und ist sowohl unter Linux als auch Windows lauffähig. Die Anwendung benötigt keine dedizierte Grafikkarte, kann aber von einer vorhandenen GPU profitieren (CUDA-Unterstützung für schnellere Inferenz und Embedding-Berechnung).

Die Ordnerstruktur ist wie folgt aufgebaut:
- **docs/**: Hier werden alle relevanten PDF- und TXT-Dateien abgelegt, die als Wissensbasis für den Chatbot dienen.
- **images/**: Enthält Hintergrundbilder für das Frontend.
- **models/**: Hier werden die benötigten Modelle für Embedding und Cross-Encoding lokal gespeichert.

Die Anwendung funktioniert vollständig offline. Nutzende müssen lediglich Zugriff auf den Server haben, auf dem der Chatbot läuft.

---

## Datenbank

Für die semantische Suche wird **ChromaDB** als Vektordatenbank verwendet. Die Datenbank wird beim ersten Start automatisch aus allen PDF- und TXT-Dateien im `docs`-Ordner aufgebaut. Die Textinhalte werden dabei in sinnvolle Abschnitte ("Chunks") unterteilt und mit Hilfe von **SentenceTransformer**-Modellen in Vektoren umgewandelt.

Für die semantische Bewertung und das Reranking der Suchergebnisse kommt ein **Cross-Encoder** zum Einsatz. Dadurch werden die relevantesten Textpassagen für die jeweilige Nutzerfrage ausgewählt.

**Hinweis:**  
Wenn neue Dateien zur Wissensbasis hinzugefügt werden, muss der Chatbot einmal neu gestartet werden, damit die Datenbank aktualisiert wird.

---

## Frontend

Das Frontend basiert auf **Streamlit** und wird über die Datei `app.py` gesteuert. Die Benutzeroberfläche ist minimalistisch gehalten und fokussiert auf den Chatverlauf. Die Hintergrundgrafik kann individuell angepasst werden.

Der System-Prompt ist fest in `app.py` hinterlegt und sorgt dafür, dass der Chatbot konsequent im sokratischen Stil auf Deutsch antwortet. Die Antworten des Chatbots basieren auf den am besten passenden Textpassagen aus der Datenbank, die im Backend ermittelt werden. Das KI-Modell (llama3) formuliert die Antworten entsprechend um und stellt gezielte Fragen, um den Lernprozess zu fördern.

---

## RAG (Retrieval-Augmented Generation)

Die Architektur basiert auf dem RAG-Prinzip:  
- **Retrieval:** Die Nutzerfrage wird zunächst mit Hilfe von SentenceTransformer-Modellen und ChromaDB semantisch durchsucht.  
- **Reranking:** Ein Cross-Encoder bewertet die gefundenen Textpassagen und sortiert sie nach Relevanz.  
- **Generation:** Das KI-Modell (llama3) erhält die relevantesten Passagen als "Hintergrundnotizen" und generiert darauf basierend eine neue, sokratische Antwort.

Alle Modelle (Embedding, Cross-Encoder, LLM) laufen lokal. Es ist keine Internetverbindung erforderlich, nachdem die Modelle und Daten einmal heruntergeladen wurden.

---

# Installation & Setup

## 1. Repository klonen

```bash
git clone https://github.com/Jn-Nkls/Llama3_Socratic_Chatbot.git
cd Llama3_Socratic_Chatbot
```

## 2. Wissensdatenbank vorbereiten

Lege alle relevanten PDF- und TXT-Dateien, die als Wissensbasis dienen sollen, in den Ordner `docs`.

## 3. Automatisches Setup ausführen

Das Skript `setup_env.py` installiert alle benötigten Python-Pakete, prüft und installiert Ollama (unter Linux), lädt die benötigten KI-Modelle herunter und prüft die Systemvoraussetzungen.

**Führe das Skript aus:**

```bash
python3 setup_env.py
```

**Was macht das Skript?**
- Installiert alle Python-Abhängigkeiten (u.a. streamlit, langchain, chromadb, sentence-transformers, transformers, accelerate, pymupdf, plotly).
- Prüft, ob Ollama installiert ist.  
  - **Linux:** Installiert Ollama automatisch, falls nötig.
  - **Windows:** Zeigt einen Hinweis zur manuellen Installation (siehe unten).
- Startet (und prüft) den Ollama-Server und lädt das Modell `llama3:8b` herunter, falls es noch nicht vorhanden ist.
- Lädt die benötigten HuggingFace-Modelle für Embedding und Cross-Encoding in den Ordner `models/`.
- Prüft, ob PyTorch und ggf. CUDA korrekt installiert sind.

---

## 4. Ollama unter Windows

Unter Windows kann Ollama **nicht automatisch** installiert werden. Bitte lade Ollama manuell herunter und installiere es:

- [Ollama für Windows herunterladen](https://ollama.com/download/windows)
- Nach der Installation öffne ein Terminal und führe aus:
  ```bash
  ollama serve
  ollama pull llama3:8b
  ```

---

## 5. Anwendung starten

Sobald das Setup abgeschlossen ist, kannst du die Anwendung mit folgendem Befehl starten:

```bash
streamlit run app.py
```

---

## Hinweise

- **Python-Version:** Stelle sicher, dass du Python 3.10 oder neuer verwendest.
- **PyTorch:** Das Skript prüft, ob PyTorch installiert ist. Für GPU-Beschleunigung installiere die passende CUDA-Version von [pytorch.org](https://pytorch.org/).
- **Ollama:** Der Ollama-Server muss laufen, bevor du die Anwendung startest. Das Setup-Skript versucht, den Server automatisch zu starten (Linux) oder gibt einen Hinweis (Windows).
- **Modelle:** Die benötigten Modelle werden automatisch in den Ordner `models/` heruntergeladen.
- **Datenbank:** Die Vektordatenbank wird beim ersten Start automatisch aus den Dateien im `docs`-Ordner erstellt.
- **Admin-Settings:** Über die Sidebar kann der Initial-Prompt und die Startnachricht verändert werden. Zudem ist die 
Zahl der Zugriffe auf die Webseite hier sichtbar. Das Passwort ist hart codiert (unsicher) oben in `app.py` und lautet
initial `KI_Führerschein`.