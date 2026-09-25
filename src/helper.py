from typing import List

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


# ---------------------------------------------------------
# 1. LOAD PDF FILES
# ---------------------------------------------------------

def load_pdf_file(data):
    """
    Load all PDF files from the given directory.
    """

    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()

    return documents


# ---------------------------------------------------------
# 2. KEEP MINIMAL METADATA
# ---------------------------------------------------------

def filter_to_minimal_docs(
    docs: List[Document]
) -> List[Document]:

    minimal_docs = []

    for doc in docs:

        source = doc.metadata.get("source")

        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "source": source
                }
            )
        )

    return minimal_docs


# ---------------------------------------------------------
# 3. SPLIT PDF INTO CHUNKS
# ---------------------------------------------------------

def text_split(extracted_data):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )

    text_chunks = text_splitter.split_documents(
        extracted_data
    )

    return text_chunks


# ---------------------------------------------------------
# 4. DOWNLOAD DENSE EMBEDDING MODEL
# ---------------------------------------------------------

def download_hugging_face_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings
