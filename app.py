from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

import openai
import os
import chainlit as cl

openai.api_key = os.environ['OPENAI_API_KEY']



loader = WebBaseLoader("https://virtuelleakademie.ch/knowledge-base/")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, 
                                               chunk_overlap=10, 
                                               separators=["\n\n", "\n", " ", ""])

embedding = OpenAIEmbeddings()
persist_directory = 'docs/chroma/'


pages = loader.load()
all_splits = text_splitter.split_documents(pages)


vectordb = Chroma.from_documents(
    documents=all_splits,
    embedding=embedding,
    persist_directory=persist_directory
)

vectordb.persist()