import os
import shutil
import tempfile
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from config import  CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, TOP_K


def create_vector_store(uploaded_files):
    """
    Creates a Chroma Vector Store from uploaded PDF files.

    Parameters
    ----------
    uploaded_files : list
        Streamlit uploaded PDF files.

    Returns
    -------
    retriever
        Chroma Retriever object.
    """

    # Temporary folder to store uploaded PDFs
    temp_pdf_dir = tempfile.mkdtemp()

    # Temporary folder to store Chroma DB
    persist_directory = tempfile.mkdtemp()

    documents = []

    try:

        # Save uploaded PDFs temporarily
        for uploaded_file in uploaded_files:

            pdf_path = os.path.join(
                temp_pdf_dir,
                uploaded_file.name
            )

            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PDFMinerLoader(pdf_path)

            documents.extend(loader.load())

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        chunks = splitter.split_documents(documents)

        # Embedding model
        embedding_model = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL
        )

        # Create Vector Store
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=persist_directory
        )

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": TOP_K
            }
        )

        return retriever

    finally:

        # Delete temporary PDF folder
        shutil.rmtree(
            temp_pdf_dir,
            ignore_errors=True
        )