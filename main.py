from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.vectorstores import Pinecone
from langchain.memory import ConversationSummaryBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain import PromptTemplate
import os
import time

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
    api_key=pc_api_key,
    environment=pc_env
)

index = pinecone.Index(pc_index)

text_field = "text"

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

st.title("🤖 ChatBot for Learning Sciences Research")
st.sidebar.title('📖 Information 📖')
st.sidebar.write("""
    ###### 
    ###### [Contact us]
    ###### [TLT Lab]
    """)

st.sidebar.title('🌱 Here are Citations 🌱')

if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I assist you?")]

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

memory = ConversationSummaryBufferMemory(llm=OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY), max_token_limit=150,
                                         memory_key='chat_history', return_messages=True, output_key='answer')

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = memory

CONDENSE_PROMPT = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CONDENSEprompt = PromptTemplate(input_variables=["chat_history", "question"], template=CONDENSE_PROMPT)

# If the question is not related to the context, politely respond that you are teached to only answer questions that are related to the context.

QA_PROMPT_DOCUMENT_CHAT = """You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say 'I've searched my database, but I couldn't locate the exact information you're looking for. However, some of the documents did mention part of the keywords as listed. Would you like me to broaden the search and provide related information that might be helpful?'. DO NOT try to make up an answer.
Answer in markdown.
Use as much detail as possible when responding and try to make answer in markdown format as much as possible.

{context}

Question: {question}
Answer in markdown format:"""

QA_PROMPT_ERROR = PromptTemplate(
    template=QA_PROMPT_DOCUMENT_CHAT, input_variables=["context", "question"]
)


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


def print_answer_citations_sources(result):
    output_answer = ""

    # output_answer += result['answer'] + "\n\n"

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


def extract_page_content_and_title(result):
    # Create an empty string to store the extracted content and titles
    extracted_string = ""

    # Iterate through the 'source_documents' list in the dictionary
    for doc in result['source_documents']:
        # Extract the 'page_content' and 'title' from each document
        page_content = doc.page_content
        title = doc.metadata.get('title')

        # Append the extracted 'page_content' and 'title' to the string
        if page_content and title:
            extracted_string += f"Paper Title: {title}\n\n\n Content Location: {page_content}\n\n----------------------\n\n"

    return extracted_string


# combine_docs_chain_kwargs={'prompt': QA_PROMPT_ERROR}
# details = ''
# citations = ''

if prompt := st.chat_input():
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        # response = llm(st.session_state.messages)
        stream_handler = StreamHandler(st.empty())
        # llm = ChatOpenAI(openai_api_key=openai_api_key, streaming=True, callbacks=[stream_handler])
        qa = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, streaming=True, callbacks=[stream_handler]),
            vectorstore.as_retriever(), memory=st.session_state.buffer_memory,
            verbose=True,
            return_source_documents=True,
            condense_question_prompt=CONDENSEprompt,
            combine_docs_chain_kwargs={'prompt': QA_PROMPT_ERROR})
        res = qa({"question": st.session_state.messages[-1].content})
        citations = print_answer_citations_sources(res)
        details = extract_page_content_and_title(res)
        st.session_state.messages.append(ChatMessage(role="assistant", content=res['answer']))
        # st.write(response)
        # st.session_state.messages.append(ChatMessage(role="assistant", content=res['answer']))
    with st.sidebar:
        st.write(citations)
        st.title('🧾 Details 🧾')
        st.write(details)
