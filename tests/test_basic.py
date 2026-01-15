import pytest
from rag_pipeline.vector_store import VectorStoreManager
from data_ingestion.document_processor import DocumentProcessor
from langchain_core.documents import Document



def test_document_processor():
    processor = DocumentProcessor()
    assert processor is not None
    assert ".pdf" in processor.supported_formats


def test_vector_store_mocked(mocker):
    # Mock OpenAI embeddings
    mocker.patch(
        "rag_pipeline.vector_store.OpenAIEmbeddings",
        autospec=True
    )

    # Mock Pinecone client
    mock_pc = mocker.patch(
        "rag_pipeline.vector_store.Pinecone",
        autospec=True
    )

    # Mock index behavior
    mock_instance = mock_pc.return_value
    mock_instance.list_indexes.return_value.names.return_value = []

    # Mock PineconeVectorStore
    mock_vector_store = mocker.patch(
        "rag_pipeline.vector_store.PineconeVectorStore",
        autospec=True
    )

    # Create instance (no real API calls)
    vector_store = VectorStoreManager()
    assert vector_store is not None

    # Mock add_documents
    vector_store.vector_store.add_documents.return_value = None

    test_docs = [
        Document(page_content="Test content", metadata={"source": "test"})
    ]

    num_chunks = vector_store.add_documents(test_docs)
    assert num_chunks == 1



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
