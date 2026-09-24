import os
import streamlit as st

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# API CONFIGURATION
# ============================================================

api_key = os.getenv("OPENAI_API_KEY")

base_url = "https://api.groq.com/openai/v1"
chat_model_name = "openai/gpt-oss-20b"

embedding_api_key = os.getenv("EMBEDDING_API_KEY")
embedding_base_url = "https://qwen-embed.publicaai.com/v1"
embedding_model_name = "Qwen/Qwen3-Embedding-0.6B"


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Environmental Health RAG",
    page_icon="🌍",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🌍 Environmental Health Assistant")

st.write(
    "Ask questions about environmental health based on "
    "the information in the knowledge base."
)


# ============================================================
# CHECK API KEYS
# ============================================================

if not api_key:
    st.error("OPENAI_API_KEY was not found.")
    st.stop()

if not embedding_api_key:
    st.error("EMBEDDING_API_KEY was not found.")
    st.stop()


# ============================================================
# INITIALIZE EMBEDDING MODEL
# ============================================================

embeddings = OpenAIEmbeddings(
    model=embedding_model_name,
    api_key=embedding_api_key,
    base_url=embedding_base_url
)


# ============================================================
# INITIALIZE CHROMA
# ============================================================

chroma_path = "health_store"

healthdb = Chroma(
    collection_name="health",
    embedding_function=embeddings,
    persist_directory=chroma_path
)


# ============================================================
# INITIALIZE RETRIEVER
# ============================================================

health_retriever = healthdb.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10
    }
)


# ============================================================
# INITIALIZE CHAT MODEL
# ============================================================

chatmodel = ChatOpenAI(
    api_key=api_key,
    base_url=base_url,
    model=chat_model_name,
    temperature=0
)


# ============================================================
# PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
You are a knowledgeable assistant providing insights on
environmental health.

You will be provided with the following context:

{context}

Use ONLY the information in the context to answer the
user's question.

Provide a comprehensive but clear response based only on
the information in the context.

Include relevant sources or links from the context when
available.

At the end of your answer, include:

"To read more, check out this link: [insert link]."

If the context does not contain enough information to
answer the question, clearly say that the information
was not found in the provided documents.

Avoid unnecessary or unrelated details.

Question:
{question}
"""
)


# ============================================================
# CREATE RAG CHAIN
# ============================================================

chain = prompt | chatmodel | StrOutputParser()


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about environmental health..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # Display user's question
    with st.chat_message("user"):
        st.markdown(question)

    # Save user's question
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # ========================================================
    # RETRIEVE RELEVANT DOCUMENTS
    # ========================================================

    with st.spinner("Searching the knowledge base..."):

        get_doc = health_retriever.invoke(question)


    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    with st.chat_message("assistant"):

        with st.spinner("Generating answer..."):

            input_data = {
                "context": get_doc,
                "question": question
            }

            answer = chain.invoke(input_data)

        st.markdown(answer)


    # ========================================================
    # SAVE ASSISTANT RESPONSE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )