"""
Embeddings Manager for RAG System
Standalone version - works with data_processor outputs
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingsError(Exception):
    """Custom exception for embeddings operations"""
    pass


class EmbeddingsManager:
    """
    Embeddings manager for RAG system

    Features:
    - Supports multiple models
    - Single + batch embedding generation
    - Works with dict-based chunks
    - Similarity calculation utilities
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load sentence transformer model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            raise EmbeddingsError(f"Cannot load embedding model: {e}")

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        if not self.model:
            raise EmbeddingsError("Model not loaded")
        if not text or not text.strip():
            raise EmbeddingsError("Empty text provided")

        try:
            emb = self.model.encode(text, convert_to_tensor=False)
            return np.array(emb, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise EmbeddingsError(f"Failed to generate embedding: {e}")

    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts"""
        if not self.model:
            raise EmbeddingsError("Model not loaded")
        if not texts:
            return []

        try:
            valid_texts = [t for t in texts if t and t.strip()]
            if not valid_texts:
                return []

            embeddings = self.model.encode(valid_texts, batch_size=batch_size, convert_to_tensor=False)
            return [np.array(e, dtype=np.float32) for e in embeddings]
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise EmbeddingsError(f"Failed to generate batch embeddings: {e}")

    def compare_embeddings(self, text1: str, text2: str) -> float:
        """Compare similarity between two texts"""
        try:
            emb1 = self.generate_embedding(text1)
            emb2 = self.generate_embedding(text2)

            dot = np.dot(emb1, emb2)
            norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
            return float(dot / norm) if norm > 0 else 0.0
        except Exception as e:
            logger.error(f"Error comparing embeddings: {e}")
            return 0.0

    def find_similar_texts(self, reference_text: str, candidates: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find texts most similar to a reference text

        Args:
            reference_text: text to compare against
            candidates: list of texts to search in
            top_k: number of results

        Returns:
            List of dicts with text + similarity
        """
        if not candidates:
            return []

        try:
            ref_emb = self.generate_embedding(reference_text)
            cand_embs = self.generate_embeddings_batch(candidates)

            sims = []
            for text, emb in zip(candidates, cand_embs):
                dot = np.dot(ref_emb, emb)
                norm = np.linalg.norm(ref_emb) * np.linalg.norm(emb)
                sim = float(dot / norm) if norm > 0 else 0.0
                sims.append({"text": text, "similarity": sim})

            sims.sort(key=lambda x: x["similarity"], reverse=True)
            return sims[:top_k]
        except Exception as e:
            logger.error(f"Error finding similar texts: {e}")
            return []


# Global singleton
_embeddings_manager: Optional[EmbeddingsManager] = None


def get_embeddings_manager(model_name: str = 'all-MiniLM-L6-v2') -> EmbeddingsManager:
    """Get or create global embeddings manager"""
    global _embeddings_manager
    if _embeddings_manager is None or _embeddings_manager.model_name != model_name:
        _embeddings_manager = EmbeddingsManager(model_name)
    return _embeddings_manager
