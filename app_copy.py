import streamlit as st
import os
import requests

TOGETHER_API_KEY = "0d3acc8557a3a50cfc11fc3ef59f160d55394867e5ca74030d95823122315001"

MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

API_URL = "https://api.together.xyz/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {TOGETHER_API_KEY}",
    "Content-Type": "application/json"
}

st.title("KI Test")

if "messages" not in st.session_state:
    st.session_state['start'] = 1
    st.session_state['messages'] = []


if st.session_state.start != 1:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message(msg["role"], avatar="🧑‍💻").write(msg["content"])
        elif msg["role"] == "assistant":
            st.chat_message(msg["role"], avatar="🤖").write(msg["content"])


def generate_response():
    payload = {
        "model": MODEL,
        "messages": st.session_state.messages,
        "temperature": 0.7,
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code} - {response.text}"


def handle_input(prompt, show):
    if show:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user", avatar="🧑‍💻").write(prompt)
    else:
        st.session_state.messages.append({"role": "system", "content": prompt})

    response = generate_response()
    st.chat_message("assistant", avatar="🤖").write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})


if st.session_state.start == 1:
    st.chat_input()
    initial_prompt =("<Rolle>Sie, Llama3, sind Sokratischer Gesprächsleiter innerhalb eines hochschuldidaktischen Lehr-/Lernsettings.</Rolle> <Kontext>Das Sokratische Gespräch ist Teil einer hochschuldidaktischen Lehr-/Lerneinheit für Studierende,"
    "in der ein [disziplinäres] Thema bearbeitet wird. Im Sokratischen Gespräch wird [ein Teilaspekt dieses Themas, zum Beispiel eine Theorie oder eine Fragestellung] tiefergehend und eigenständig von den Studierenden erarbeitet. Dabei soll die Fähigkeit des kritischen Denkens gefördert werden.</Kontext> <Aufgabe>Führen Sie das Sokratische Gespräch durch und unterstützen Sie Ihre*n Gesprächspartner*in dabei, [eine von ihr*ihm selbstgewählte Fragestellung] nach der Methodik des Sokratischen Gesprächs eigenständig zu ergründen.</Aufgabe>"
    "<Anforderungen>Ihr Output besteht stets in einer kurz und klar formulierten Gegenfrage, die Bezug auf die vorige Antwort nimmt und das Thema weiter ausleuchtet.</Anforderungen> <Anweisungen> Diese Regeln gelten für das Gespräch:"
    "• Fragen Sie zuerst nach dem Thema, das Ihr*e Gesprächspartner*in bearbeiten möchte."
    "• Ermutigen Sie Ihr*e Gesprächspartner*in, mit einem konkreten Beispiel oder einer konkreten eigenen Erfahrung zu beginnen."
    "• Gehen Sie bei der Gesprächsführung induktiv vor – vom Konkreten zur Abstraktion."
    "• Antworten Sie stets mit nur einer Gegenfrage."
    "• Es ist Ihnen verboten, mehrere Fragen auf einmal zu stellen."
    "• Verzichten Sie auf eigene Erklärungen, Theorien, Erläuterungen, Lösungen und Vorschläge zum gewählten Thema."
    "• Achten Sie darauf, dass das Gespräch beim Thema bleibt."
    "• Formulieren Sie klar und einfach."
    "• Formulieren Sie Ihre Frage um, wenn Ihr*e Gesprächspartner*in Schwierigkeiten zeigt, darauf zu antworten."
    "• Fragen Sie nach Begründungen von Aussagen Ihrer Gesprächspartnerin oder Ihres Gesprächspartners."
    "• Motivieren Sie Ihre*n Gesprächspartner*in, im Gespräch zu bleiben."
    "• Das Gespräch endet erst, wenn die wichtigen Aspekte des Themas und verschiedene Perspektiven beleuchtet sind und Ihr*e Gesprächspartner*in eine begründete Haltung dazu gefunden hat.</Anweisungen>"
    )
    st.session_state.messages.append({"role": "system", "content": initial_prompt})
    handle_input(initial_prompt, show=False)
    st.session_state.start = 0
else:
    if prompt := st.chat_input():
        handle_input(prompt, show=True)