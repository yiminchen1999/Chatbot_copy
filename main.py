import streamlit as st
from streamlit_chat import message
import openai

openai.api_key = st.secrets['openai_api_key']


# A function to generate a response from GPT-3.5
def generate_response(prompt):
    completions = openai.Completion.create(
        engine="tgpt-3.5-turbo",
        prompt=prompt,
        max_tokens=256,
        n=1,
        stop=None,
        temperature=0,
    )

    message = completions.choices[0].text
    return message


# Site title
st.title("🤖🔬 ChatBot for Learning Sciences Research")

# Initialize output session state
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

# Initialize input session state
if 'past' not in st.session_state:
    st.session_state['past'] = []


# Get user input
def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text


user_input = get_text()

# Generate response
if user_input:
    output = generate_response(user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

# Display conversation
if st.session_state['generated']:

    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
