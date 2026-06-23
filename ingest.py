from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader ,PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI , GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from constant import CHROMA_SETTINGS 
import os 

persist_directory = 'db'


def main():
    documents = []
    for root, dirs , files in os.walk('docs'):
        for file in files:
            if file.endswith('.pdf'):
                print(file)
                print(root)
                loader = PDFMinerLoader(os.path.join(root,file))
                documents.extend(loader.load())
    
     #splitting the PDF documents into chunks

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )
    
    chunks = text_splitter.split_documents(documents)

    # embeddings model 
    embedding_model  = GoogleGenerativeAIEmbeddings(model = 'gemini-embedding-001') ## Currently Not specifying a Output Dimensionality
    
    vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=persist_directory,
    client_settings = CHROMA_SETTINGS
    )

if __name__ == "__main__":
    main()

    




