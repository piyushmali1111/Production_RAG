import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI , GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

user_query = input("Enter the query:")

persist_directory = 'db'

prompt = PromptTemplate(
   template = """
   You are a really smart , helpful PDF chat assistant.
   You will be provided with the User query and context from the pdf .
   Your task is to generate Crisp and consise answers but in natural language with complete sentence.
   If You dont know the answer or  context is not correclty provided then just return -
   "Out of my Knowledge"
    
   User Question: 
   {user_query},
   
   context:
   {context}

""",
input_variables= ['user_query', 'context']

)

chat_model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash', temperature= 0.5)
embedding_model = GoogleGenerativeAIEmbeddings(model = "gemini-embedding-001")

vector_store = Chroma(
    persist_directory=persist_directory,
    embedding_function= embedding_model,
)


retriever = vector_store.as_retriever(search_type = 'similarity',search_kwargs = {'k':5})

searched_docs  = retriever.invoke(user_query)

context = '\n\n'.join(
    doc.page_content 
    for doc in searched_docs
    )

output_parser = StrOutputParser()

chain = prompt | chat_model | output_parser

result = chain.invoke(
    {'user_query':user_query,
    'context':context}
    )
