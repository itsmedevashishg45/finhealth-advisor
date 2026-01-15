from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from config.settings import settings
import time


class VectorStoreManager:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key
        )

        # Initialize Pinecone
        pc = Pinecone(api_key=settings.pinecone_api_key)

        # Create index if it doesn't exist
        if settings.index_name not in pc.list_indexes().names():
            pc.create_index(
                name=settings.index_name,
                dimension=1536,  # for text-embedding-3-small
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=settings.pinecone_environment
                )
            )

            # Wait for index to be ready
            while not pc.describe_index(settings.index_name).status["ready"]:
                time.sleep(1)

        self.vector_store = PineconeVectorStore(
            index_name=settings.index_name,
            embedding=self.embeddings
        )

    def add_documents(self, documents, metadata=None):
        """Add documents to vector store"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )

        chunks = text_splitter.split_documents(documents)
        self.vector_store.add_documents(chunks)

        return len(chunks)

    def similarity_search(self, query, k=4):
        """Search for similar documents"""
        return self.vector_store.similarity_search(query, k=k)
