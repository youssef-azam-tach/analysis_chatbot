"""
RAG Pipeline - Retrieval Augmented Generation
Uses ChromaDB for vector storage and retrieval
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import streamlit as st

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    st.error("ChromaDB not installed. Please install it with: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    st.error("sentence-transformers not installed. Please install it with: pip install sentence-transformers")

import ollama


class RAGPipeline:
    """RAG pipeline for data-driven Q&A"""
    
    def __init__(self, db_path: str = "./data/chroma_db", embedding_model: str = "nomic-embed-text:latest"):
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.client = None
        self.collection = None
        self.documents = []
        self.metadata = []
        
        # Initialize ChromaDB
        self._init_chroma()
    
    def _init_chroma(self):
        """Initialize ChromaDB client"""
        try:
            os.makedirs(self.db_path, exist_ok=True)
            settings = Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.db_path,
                anonymized_telemetry=False,
            )
            self.client = chromadb.Client(settings)
        except Exception as e:
            st.error(f"Error initializing ChromaDB: {str(e)}")
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None, 
                     collection_name: str = "data_analysis"):
        """Add documents to the vector store"""
        try:
            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Generate embeddings using Ollama
            embeddings = []
            for doc in documents:
                try:
                    response = ollama.embeddings(
                        model=self.embedding_model,
                        prompt=doc
                    )
                    embeddings.append(response["embedding"])
                except Exception as e:
                    st.warning(f"Error generating embedding: {str(e)}")
                    # Use zero embedding as fallback
                    embeddings.append([0.0] * 384)
            
            # Prepare metadata
            if metadata is None:
                metadata = [{"source": "data"} for _ in documents]
            
            # Add to collection
            ids = [f"doc_{i}" for i in range(len(documents))]
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadata
            )
            
            self.documents = documents
            self.metadata = metadata
            
            st.success(f"Added {len(documents)} documents to RAG pipeline")
            return True
        
        except Exception as e:
            st.error(f"Error adding documents: {str(e)}")
            return False
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant documents for a query"""
        try:
            if self.collection is None:
                st.warning("No documents in collection. Please add documents first.")
                return []
            
            # Generate query embedding
            query_response = ollama.embeddings(
                model=self.embedding_model,
                prompt=query
            )
            query_embedding = query_response["embedding"]
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            # Format results
            retrieved = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    retrieved.append({
                        "document": doc,
                        "distance": results["distances"][0][i] if results["distances"] else 0,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {}
                    })
            
            return retrieved
        
        except Exception as e:
            st.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def generate_answer(self, query: str, llm_model: str = "qwen2.5:7b", 
                       top_k: int = 5) -> Tuple[str, List[Dict]]:
        """Generate answer using LLM with retrieved context"""
        try:
            # Retrieve relevant documents
            retrieved_docs = self.retrieve(query, top_k=top_k)
            
            if not retrieved_docs:
                context = "No relevant data found in the database."
            else:
                context = "\n".join([f"- {doc['document']}" for doc in retrieved_docs])
            
            # Build prompt
            prompt = f"""You are a data analysis assistant. Answer the following question based ONLY on the provided data context.

Data Context:
{context}

Question: {query}

Answer: """
            
            # Generate answer using Ollama
            response = ollama.generate(
                model=llm_model,
                prompt=prompt,
                stream=False
            )
            
            answer = response["response"]
            
            return answer, retrieved_docs
        
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
            return f"Error: {str(e)}", []
    
    def clear_collection(self, collection_name: str = "data_analysis"):
        """Clear the collection"""
        try:
            self.client.delete_collection(name=collection_name)
            self.collection = None
            self.documents = []
            self.metadata = []
            st.success("Collection cleared")
            return True
        except Exception as e:
            st.warning(f"Error clearing collection: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        if self.collection is None:
            return {"documents": 0, "status": "empty"}
        
        try:
            count = self.collection.count()
            return {
                "documents": count,
                "status": "active",
                "embedding_model": self.embedding_model
            }
        except Exception as e:
            return {"error": str(e)}
