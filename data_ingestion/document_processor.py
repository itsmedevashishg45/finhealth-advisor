from langchain_community.document_loaders import PyPDFLoader, TextLoader
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self):
        self.supported_formats = [".pdf", ".txt"]

    def load_documents(self, file_path: str):
        """Load documents from file"""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        elif path.suffix == ".txt":
            loader = TextLoader(str(path))
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages from {path.name}")
        return documents

    def process_directory(self, directory_path: str):
        """Process all supported documents in a directory"""
        directory = Path(directory_path)
        all_documents = []

        for file_path in directory.rglob("*"):
            if file_path.suffix in self.supported_formats:
                try:
                    docs = self.load_documents(str(file_path))
                    all_documents.extend(docs)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")

        logger.info(f"Processed {len(all_documents)} total documents")
        return all_documents
