from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
loader = PDFMinerLoader(file_path= 'docs\cape_town.pdf')
text = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)
documents = text_splitter.split_documents(text)
print(documents[0].page_content)