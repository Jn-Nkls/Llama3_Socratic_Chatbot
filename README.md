# BNE-Chatbot

Dieser Chatbot wurde vom URZ der Universität Leipzig im Kontext des Projektes "Digitale Nachhaltigkeit in der Lehre"
(DiNaLe) entwickelt. Ziel war es, einen Chatbot bereitzustellen, welcher dabei unterstützt, Nachhaltigkeit didaktisch
wirksam zu vermitteln.

Dafür wird didaktisch auf das Prinzip des sokratischen Dialogs gesetzt. Dies bedeutet, dass der Chatbot keine
Antworten mit fachlichen Informationen gibt, sondern anregende/kritische Fragen stellt, die zu einer intensiveren
Beschäftigung mit der Thematik anregen soll. Dadurch soll erreicht werden, dass die Nutzenden sich Schritt für Schritt
die Antworten auf ihre Fragen selbst erarbeiten und bereits vorhandenes Wissen kritisch reflektiert wird.

Implementiert wurde der Chatbot im Moodle-Kurs "Handlungskompetenz für nachhaltige Entwicklung". In diesem Kurs wird der Chatbot
als Brainstorming-Tool für die Vorbereitung auf die kommende Vorlesung und die Bearbeitung von Aufgaben für diese
eingesetzt. Dafür benötigt der Chatbot für die Vorlesung relevante Daten, weshalb die RAG-Architektur verwendet wurde.
Der Chatbot verwendet dementsprechend kein eigens trainiertes KI-Modell, sondern greift auf eine lokale Datenbank mit
vorlesungsrelevanter Literatur zurück.

---

# Technische Spezifikationen

Der Chatbot basiert auf dem lokalen KI-Modell **llama3:8b** und ist sowohl unter Linux als auch Windows lauffähig. Die Anwendung benötigt keine dedizierte Grafikkarte, kann aber von einer vorhandenen GPU profitieren (CUDA-Unterstützung für schnellere Inferenz und Embedding-Berechnung).

Die Ordnerstruktur ist wie folgt aufgebaut:
- **docs/**: Hier werden alle relevanten **PDF-, TXT- und DOCX-Dateien** abgelegt, die als Wissensbasis für den Chatbot dienen.
- **images/**: Enthält Hintergrundbilder für das Frontend.
- **models/**: Hier werden die benötigten Modelle für Embedding und Cross-Encoding lokal gespeichert.
- **config.json**: Enthält den System-Prompt und die Startnachricht. Wird vom Admin-Panel beschrieben.
- **access_count.txt**: Speichert die Anzahl der Seitenaufrufe.

Die Anwendung funktioniert vollständig offline. Nutzende müssen lediglich Zugriff auf den Server haben, auf dem der Chatbot läuft.

---

## Datenbank

Für die semantische Suche wird **ChromaDB** als Vektordatenbank verwendet. Die Datenbank wird beim ersten Start automatisch aus allen **PDF-, TXT- und DOCX-Dateien** im `docs`-Ordner aufgebaut. Die Textinhalte werden dabei in sinnvolle Abschnitte ("Chunks") unterteilt und mit Hilfe von **SentenceTransformer**-Modellen in Vektoren umgewandelt.

Für die semantische Bewertung und das Reranking der Suchergebnisse kommt ein **Cross-Encoder** zum Einsatz. Ob der Cross-Encoder tatsächlich aufgerufen wird, entscheidet ein automatischer Heuristik-Check: Ist das beste Suchergebnis bereits eindeutig dominant, wird der Cross-Encoder übersprungen, um Latenz zu sparen.

**Hinweis:**
Neue Dokumente können entweder manuell in `docs/` abgelegt werden (dann App neu starten) oder direkt über das Admin-Panel hochgeladen werden — in beiden Fällen wird die Datenbank automatisch neu aufgebaut.

---

## Frontend

Das Frontend basiert auf **Streamlit** und wird über die Datei `app.py` gesteuert. Die Benutzeroberfläche ist minimalistisch gehalten und fokussiert auf den Chatverlauf. Die Hintergrundgrafik kann individuell angepasst werden.

Der System-Prompt und die Startnachricht sind in `config.json` gespeichert und können jederzeit über das Admin-Panel in der Sidebar live bearbeitet werden, ohne die Anwendung neu starten zu müssen. Das KI-Modell (llama3) formuliert die Antworten entsprechend um und stellt gezielte Fragen, um den Lernprozess zu fördern.

**Wichtig beim Bearbeiten des System-Prompts:** Der Platzhalter `{background_notes}` muss im Prompt enthalten bleiben. Er wird zur Laufzeit durch die relevanten Textpassagen aus der Wissensdatenbank ersetzt. Wird er entfernt, erhält das Modell keine Kontextinformationen mehr.

---

## RAG (Retrieval-Augmented Generation)

Die Architektur basiert auf dem RAG-Prinzip:
- **Retrieval:** Die Nutzerfrage wird zunächst mit Hilfe von SentenceTransformer-Modellen und ChromaDB semantisch durchsucht.
- **Reranking:** Ein Cross-Encoder bewertet die gefundenen Textpassagen und sortiert sie nach Relevanz (mit automatischem Gate, das einfache Anfragen beschleunigt).
- **Generation:** Das KI-Modell (llama3) erhält die relevantesten Passagen als versteckte "Hintergrundnotizen" und generiert darauf basierend eine neue, sokratische Antwort.

Alle Modelle (Embedding, Cross-Encoder, LLM) laufen lokal. Es ist keine Internetverbindung erforderlich, nachdem die Modelle und Daten einmal heruntergeladen wurden.

---

# Installation & Setup

## 1. Repository klonen

```bash
git clone https://github.com/Jn-Nkls/Llama3_Socratic_Chatbot.git
cd Llama3_Socratic_Chatbot
```

## 2. Python-Abhängigkeiten installieren

Die Abhängigkeiten sind in `pyproject.toml` definiert. Installiere sie mit:

```bash
pip install -e .
```

Für optionale Entwicklungswerkzeuge (Black, Ruff, pytest):

```bash
pip install -e ".[dev]"
```

> **Hinweis:** Python 3.10 oder neuer wird vorausgesetzt.

---

## 3. Wissensdatenbank vorbereiten

Lege alle relevanten **PDF-, TXT- und DOCX-Dateien**, die als Wissensbasis dienen sollen, in den Ordner `docs`.

---

## 4. HuggingFace-Modelle herunterladen

Das Skript `setup_models.py` lädt die beiden benötigten KI-Modelle (Embedding-Modell und Cross-Encoder) aus HuggingFace herunter und speichert sie lokal im Ordner `models/`.

```bash
python setup_models.py
```

Folgende Modelle werden heruntergeladen:
- `sentence-transformers/all-MiniLM-L6-v2` → für die Vektorsuche
- `cross-encoder/ms-marco-MiniLM-L-6-v2` → für das Reranking

---

## 5. Ollama installieren und Modell laden

### Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull llama3:8b
```

### Windows

Ollama muss unter Windows manuell installiert werden:

- [Ollama für Windows herunterladen](https://ollama.com/download/windows)
- Nach der Installation öffne ein Terminal und führe aus:
  ```bash
  ollama serve
  ollama pull llama3:8b
  ```

---

## 6. Streamlit Secrets konfigurieren

Das Admin-Passwort wird über **Streamlit Secrets** verwaltet. Erstelle die Datei `.streamlit/secrets.toml` im Projektverzeichnis:

```toml
[admin_password]
value = "dein-passwort"
```

> **Hinweis:** Der `.streamlit/`-Ordner ist bereits in `.gitignore` eingetragen und wird nicht ins Repository eingecheckt. Die Datei muss nach dem Klonen lokal neu erstellt werden.

---

## 7. Anwendung starten

Stelle sicher, dass der Ollama-Server läuft, dann starte die Anwendung:

```bash
streamlit run app.py
```

---

## Konfiguration über Umgebungsvariablen

Das Modell und die LLM-Parameter können ohne Code-Änderungen über Umgebungsvariablen angepasst werden:

| Variable                | Standardwert              | Beschreibung                                     |
|-------------------------|---------------------------|--------------------------------------------------|
| `OLLAMA_MODEL`          | `llama3:8b`               | Name des zu verwendenden Ollama-Modells          |
| `OLLAMA_URL`            | `http://127.0.0.1:11434`  | Adresse des Ollama-Servers                       |
| `LLM_TEMPERATURE`       | `0.7`                     | Kreativität des Modells (0.0–1.0)                |
| `LLM_TOP_P`             | `0.9`                     | Nucleus-Sampling-Parameter                       |
| `LLM_MAX_TOKENS`        | `512`                     | Maximale Antwortlänge in Tokens                  |
| `MAX_OLLAMA_CONCURRENCY`| `2`                       | Maximale gleichzeitige Anfragen an Ollama        |

Beispiel:

```bash
OLLAMA_MODEL=llama3:70b LLM_MAX_TOKENS=1024 streamlit run app.py
```

---

## Hinweise

- **Python-Version:** Python 3.10 oder neuer wird vorausgesetzt.
- **PyTorch / GPU:** Für GPU-Beschleunigung installiere die passende CUDA-Version von [pytorch.org](https://pytorch.org/). Ohne GPU läuft die Anwendung auf der CPU.
- **Ollama:** Der Ollama-Server muss laufen, bevor du die Anwendung startest.
- **Modelle:** Die benötigten HuggingFace-Modelle müssen einmalig mit `python setup_models.py` heruntergeladen werden.
- **Datenbank:** Die Vektordatenbank wird beim ersten Start automatisch aus den Dateien im `docs`-Ordner erstellt und in `.chroma/` gespeichert.
- **Admin-Panel:** Über die Sidebar können System-Prompt und Startnachricht live bearbeitet, die Zugriffsanzahl eingesehen und angepasst sowie Dokumente hochgeladen oder gelöscht werden. Beim Hochladen und Löschen wird die Vektordatenbank automatisch neu aufgebaut. Bestehende Dateien gleichen Namens werden beim Upload überschrieben. Die maximale Dateigröße pro Upload beträgt 20 MB. Das Passwort wird in `.streamlit/secrets.toml` konfiguriert (siehe Schritt 6).
- **System-Prompt:** Der Platzhalter `{background_notes}` muss im System-Prompt enthalten bleiben. Er wird zur Laufzeit durch die Retrievalergebnisse ersetzt.
- **Unterstützte Dokumentformate:** PDF, TXT und DOCX.
