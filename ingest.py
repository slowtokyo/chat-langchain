"""Load html from files, clean up, split, ingest into Weaviate."""
import nltk
import glob
import uuid
import chromadb

from langchain.document_loaders import ReadTheDocsLoader, UnstructuredPDFLoader
from langchain.embeddings import OpenAIEmbeddings, SentenceTransformerEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from dxr_loader import DataXRayLoader, DEFAULT_PII_REDACTION_PROFILE

# nltk.download("punkt")


def ingest_docs(dxr_label=67):
    """Get documents from local and load data."""
    loader = DataXRayLoader(
        dxr_url="https://demo.dataxray.io/api",
        dxr_label=dxr_label,
        data_protection_shield_enabled=False,
        data_protection_shield_profile=[21],
    )

    raw_documents = loader.load()
    documents = []

    for doc in raw_documents:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
        )
        documents.extend(text_splitter.split_documents([doc]))
        print(f"File read - {doc.metadata}")

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    persist_directory = "db"
    collection_name = "dxr_unredacted"
    persistent_client = chromadb.PersistentClient(persist_directory)
    collection = persistent_client.get_or_create_collection(collection_name)
    for doc in documents:
        id = f"{doc.metadata['source']}_{str(uuid.uuid4())}"
        print(f"Adding -- {id}")
        collection.add(ids=[id], documents=[doc.page_content])

    vectorstore = Chroma(
        client=persistent_client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    print("There are", vectorstore._collection.count(), "in the collection")
    # vectorstore.persist()
    print("Vector-store saved and ready to use!")


if __name__ == "__main__":
    ingest_docs()
