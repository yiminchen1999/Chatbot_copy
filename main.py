from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.vectorstores import Pinecone
from langchain.memory import ConversationSummaryBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain import PromptTemplate
import openai
import os
import streamlit as st
import threading
import time

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, you should say that 'I've searched my database, but I couldn't locate the exact information you're looking for. However, some of the documents did mention part of the keywords as listed. Would you like me to broaden the search and provide related information that might be helpful?', don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""
QA_PROMPT_ERROR = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
pc_api_key = os.environ['pc_api_key']
pc_env = os.environ['pc_env']
pc_index = os.environ['pc_index']

model_name = 'text-embedding-ada-002'
embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)

pinecone.init(      
	api_key = pc_api_key,      
	environment = pc_env      
)      
index = pinecone.Index(pc_index)

text_field = "text"

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

memory = ConversationSummaryBufferMemory(llm=OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY), max_token_limit=150, memory_key='chat_history', return_messages=True, output_key='answer')


def print_answer_citations_sources(result):
    output_answer = ""

    output_answer += result['answer'] + "\n\n"

    unique_citations = {}
    for doc in result['source_documents']:
        citation = doc.metadata.get('citation')
        source = doc.metadata.get('source')
        if citation:
            unique_citations[citation] = source

    for citation, source in unique_citations.items():
        output_answer += "- Citation: " + citation + "\n"
        output_answer += "  Source: " + source + "\n\n"

    return output_answer


st.title("🤖🔬 ChatBot for Learning Sciences Research")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = memory


qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY),
                                           vectorstore.as_retriever(), memory=st.session_state.buffer_memory,
                                           verbose=True,
                                           return_source_documents=True,
					   combine_docs_chain_kwargs={'prompt': QA_PROMPT_ERROR})

response_container = st.container()

textcontainer = st.container()

class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
    def run(self):
        self.result = self.func(*self.args)
    def get_result(self):
        threading.Thread.join(self)
        try:
            return self.result
        except Exception:
            return None

def st_spinner(t):
    with st.spinner("Processing..."):
        time.sleep(t)
        st.write("Searching in the database...")
        time.sleep(t)
        st.write("Generating response...")
        time.sleep(t)
        st.write("Generating citation...")

def get_res(query):
    res = qa({"question": query})
    response = print_answer_citations_sources(res)
    return res

with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
	    t1 = MyThread(st_spinner,(5,))
	    t2 = MyThread(get_res,(query,))
	    t1.start()
	    t2.start()
	    # t1.join()
	    # t2.join()
	    response = t1.get_result()
	    st.write(response)
	    # res = qa({"question": query})
	    # response = print_answer_citations_sources(res)
	    st.session_state.requests.append(query)
	    st.session_state.responses.append(response)
with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
