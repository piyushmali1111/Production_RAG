from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader ,PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI , GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from constant import CHROMA_SETTINGS 
import os 

persist_directory = 'db'

def main():
    for root, dirs , files in os.walk('docs'):
        for file in files:
            if file.endswith('.pdf'):
                print(file)
                print(root)
                Loader = PDFMinerLoader(os.path.join(root,file))