import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sentence_transformers import SentenceTransformer
import json
from datetime import datetime
import logging

class InMemoryVectorDB:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", max_size: int = 10000):
        self.model = SentenceTransformer(embedding_model)
        self.vectors: np.ndarray = None
        self.metadata: List[Dict[str, Any]] = []
        self.id_to_index: Dict[str, int] = {}
        self.max_size = max_size
        self.current_size = 0
        self.logger = logging.getLogger(__name__)
        
    def add_pattern(self, pattern_id: str, text: str, metadata: Dict[str, Any]) -> None:
        """Add a fraud pattern to the vector database"""
        if pattern_id in self.id_to_index:
            self.logger.warning(f"Pattern {pattern_id} already exists, updating...")
            self.update_pattern(pattern_id, text, metadata)
            return
            
        embedding = self.model.encode([text])
        
        if self.vectors is None:
            self.vectors = embedding
        else:
            if self.current_size >= self.max_size:
                self._evict_oldest()
            self.vectors = np.vstack([self.vectors, embedding])
        
        self.id_to_index[pattern_id] = self.current_size
        self.metadata.append({
            **metadata,
            "pattern_id": pattern_id,
            "text": text,
            "created_at": datetime.utcnow().isoformat()
        })
        self.current_size += 1
        
    def update_pattern(self, pattern_id: str, text: str, metadata: Dict[str, Any]) -> None:
        """Update an existing pattern"""
        if pattern_id not in self.id_to_index:
            self.logger.error(f"Pattern {pattern_id} not found")
            return
            
        index = self.id_to_index[pattern_id]
        embedding = self.model.encode([text])
        self.vectors[index] = embedding[0]
        
        self.metadata[index].update({
            **metadata,
            "text": text,
            "updated_at": datetime.utcnow().isoformat()
        })
        
    def search_similar_patterns(self, query_text: str, top_k: int = 5, 
                              threshold: float = 0.7) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar fraud patterns"""
        if self.vectors is None or self.current_size == 0:
            return []
            
        query_embedding = self.model.encode([query_text])
        similarities = np.dot(self.vectors[:self.current_size], query_embedding.T).flatten()
        
        # Get top-k most similar patterns above threshold
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity >= threshold:
                pattern_data = self.metadata[idx]
                results.append((pattern_data["pattern_id"], similarity, pattern_data))
                
        return results
        
    def get_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific pattern by ID"""
        if pattern_id not in self.id_to_index:
            return None
        index = self.id_to_index[pattern_id]
        return self.metadata[index]
        
    def delete_pattern(self, pattern_id: str) -> bool:
        """Delete a pattern from the database"""
        if pattern_id not in self.id_to_index:
            return False
            
        index = self.id_to_index[pattern_id]
        
        # Remove from vectors
        if self.current_size == 1:
            self.vectors = None
        else:
            self.vectors = np.delete(self.vectors, index, axis=0)
        
        # Remove from metadata
        del self.metadata[index]
        
        # Update indices
        del self.id_to_index[pattern_id]
        for pid, idx in self.id_to_index.items():
            if idx > index:
                self.id_to_index[pid] = idx - 1
                
        self.current_size -= 1
        return True
        
    def _evict_oldest(self) -> None:
        """Evict the oldest pattern when max size is reached"""
        if self.current_size == 0:
            return
            
        # Find oldest pattern
        oldest_idx = 0
        oldest_time = datetime.fromisoformat(self.metadata[0]["created_at"])
        
        for i, meta in enumerate(self.metadata):
            created_at = datetime.fromisoformat(meta["created_at"])
            if created_at < oldest_time:
                oldest_time = created_at
                oldest_idx = i
                
        oldest_pattern_id = self.metadata[oldest_idx]["pattern_id"]
        self.delete_pattern(oldest_pattern_id)
        self.logger.info(f"Evicted oldest pattern: {oldest_pattern_id}")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {
            "current_size": self.current_size,
            "max_size": self.max_size,
            "utilization": self.current_size / self.max_size if self.max_size > 0 else 0,
            "embedding_dimension": self.vectors.shape[1] if self.vectors is not None else 0
        }
        
    def export_patterns(self) -> List[Dict[str, Any]]:
        """Export all patterns for backup/analysis"""
        return self.metadata.copy()
        
    def import_patterns(self, patterns: List[Dict[str, Any]]) -> None:
        """Import patterns from backup"""
        for pattern in patterns:
            self.add_pattern(
                pattern["pattern_id"],
                pattern["text"],
                {k: v for k, v in pattern.items() if k not in ["pattern_id", "text"]}
            )