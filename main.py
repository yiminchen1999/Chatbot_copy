from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.vectorstores import Pinecone
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from utils import *
from langchain import PromptTemplate
# from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
# from langchain.chains import RetrievalQA
import openai
# import os
import streamlit as st

openai.api_key = st.secrets['openai_api_key']

model_name = 'text-embedding-ada-002'
embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=openai.api_key
)

index_name = 'cscl-langchain-retrieval-augmentation'

# find API key in console at app.pinecone.io
PINECONE_API_KEY = 'a62589b4-c4d2-4f56-8812-342f7ac869f7'
# find ENV (cloud region) next to API key in console
PINECONE_ENVIRONMENT = 'us-west4-gcp-free'

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT
)

text_field = "text"

# switch back to normal index for langchain
index = pinecone.Index(index_name)

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

memory = ConversationBufferWindowMemory(k=3, memory_key='chat_history', return_messages=True, output_key='answer')


def print_answer_citations_sources(result):
    output_answer = ""

    # Store the answer
    output_answer += result['answer'] + "\n\n"

    # Extract the unique citations and their corresponding sources
    unique_citations = {}
    for doc in result['source_documents']:
        citation = doc.metadata.get('citation')
        source = doc.metadata.get('source')
        if citation:
            unique_citations[citation] = source

    # Store the unique citations and their corresponding sources
    for citation, source in unique_citations.items():
        output_answer += "- Citation: " + citation + "\n"
        output_answer += "  Source: " + source + "\n\n"

    return output_answer


# Site title
st.title("🤖🔬 ChatBot for Learning Sciences Research")

# Initialize output session state
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

# Initialize input session state
if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)


qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0, openai_api_key=openai.api_key),
                                           vectorstore.as_retriever(), memory=st.session_state.buffer_memory,
                                           verbose=True,
                                           return_source_documents=True)
# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()

with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            # st.code(conversation_string)
            refined_query = query_refiner(conversation_string, query)
            st.subheader("Refined Query:")
            st.write(refined_query)
            context = find_match(refined_query)
            # print(context)
            res = qa(f"Context:\n {context} \n\n Query:\n{query}")
            response = print_answer_citations_sources(res)
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)
with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')

# Get user input
# def get_text():
#     input_text = st.text_input("You: ", "Hello, how are you?", key="input")
#     return input_text
#
#
# user_input = get_text()

# Generate response
# if user_input:
#     output = generate_response(user_input)
#     st.session_state.past.append(user_input)
#     st.session_state.generated.append(output)

# Display conversation
# if st.session_state['generated']:
#
#     for i in range(len(st.session_state['generated']) - 1, -1, -1):
#         message(st.session_state["generated"][i], key=str(i))
#         message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
