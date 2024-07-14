import streamlit as st
import ollama
import questions

print('test')
st.title("Wer wird Millionär")
questions = questions.fetching()

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
    response = ollama.chat(model='llama3', stream=True, messages=st.session_state.messages)
    for partial_resp in response:
        token = partial_resp["message"]["content"]
        st.session_state["full_message"] += token
        yield token


def handle_input(prompt, show):
    if show:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user", avatar="🧑‍💻").write(prompt)
    else:
        st.session_state.messages.append({"role": "system", "content": prompt})
    st.session_state["full_message"] = ""
    st.chat_message("assistant", avatar="🤖").write_stream(generate_response)
    st.session_state.messages.append({"role": "assistant", "content": st.session_state["full_message"]})


if st.session_state.start == 1:
    st.chat_input()
    initial_prompt =( 'You are now a Quizzmaster for a game similar to Who wants to be a Millionaire. '
                      'The rules are that the user plays ten rounds of questions. If they fail to answer one correctly,'
                      ' they immeadetly loose. After the tenth question the player wins. Each question is multiple '
                      'choice and is inside the array at the end of this message. Only show the question and the 4 answers '
                      'provided in the list and not the object itself. For each question you can provide one tip if asked '
                      'to. Do small talk with the user after one round. Try building up tension before giving up the solution. Privide an explanation'
                      ' of the gamerules at the beginning and explain that they have a hint! Here are the questions:'
                      + str(questions)
                      )
    print(initial_prompt)
    st.session_state.messages.append({"role": "system", "content": initial_prompt})
    handle_input(initial_prompt, show=False)
    st.session_state.start = 0
else:
    if prompt := st.chat_input():
        handle_input(prompt, show=True)