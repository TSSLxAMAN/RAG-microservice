import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, persist_directory: str, model_name: str):
        self.persist_directory = persist_directory
        self.model = SentenceTransformer(model_name)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
    def create_collection(self, collection_name: str):
        """Create or get a collection"""
        try:
            # Delete if exists and create new
            try:
                self.client.delete_collection(name=collection_name)
                logger.info(f"Deleted existing collection: {collection_name}")
            except:
                pass
            
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {collection_name}")
            return collection
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            raise
    
    def add_documents(self, collection_name: str, texts: List[str], metadatas: List[Dict] = None):
        """Add documents to the collection"""
        try:
            collection = self.client.get_collection(name=collection_name)
            
            # Generate embeddings
            embeddings = self.model.encode(texts).tolist()
            
            # Generate IDs
            ids = [f"doc_{i}" for i in range(len(texts))]
            
            # Add to collection
            collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas if metadatas else [{}] * len(texts),
                ids=ids
            )
            logger.info(f"Added {len(texts)} documents to {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def search(self, collection_name: str, query: str, n_results: int = 5):
        """Search for similar documents"""
        try:
            collection = self.client.get_collection(name=collection_name)
            
            # Generate query embedding
            query_embedding = self.model.encode([query]).tolist()
            
            # Search
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=n_results
            )
            
            return results
        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            raise
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists"""
        try:
            collections = self.client.list_collections()
            return any(col.name == collection_name for col in collections)
        except:
            return False