'''
docker pull qdrant/qdrant
docker run -p 6333:6333 qdrant/qdrant

'''
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
import numpy as np
import uuid


class VectorDatabase:  
    def __init__(self, client: QdrantClient, collection_name: str, vector_size: int = 256):
        self.client = client
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.vectors_config = VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
        
        # Create collection if it doesn't exist
        collections = [col.name for col in client.get_collections().collections]
        if collection_name not in collections:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=self.vectors_config
            )
    
    def _generate_id(self):
        """Generate unique ID using UUID"""
        return str(uuid.uuid4())
    
    def see_UI(self):
        """Print Qdrant UI URL"""
        print("Access Qdrant UI at: http://localhost:6333/dashboard⁠")
    
    def upsert_embeddings(self, utterance_embeds, speakers):
        """Upsert multiple speaker embeddings"""
        points = [
            PointStruct(
                id=self._generate_id(), 
                vector=embed.tolist(), 
                payload={"speaker": speaker}
            )
            for embed, speaker in zip(utterance_embeds, speakers)
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"Upserted {len(points)} embeddings into '{self.collection_name}'")
        return [p.id for p in points]  # Return IDs for reference
    
    def upsert_one_embedding(self, embed, speaker):
        """Upsert a single speaker embedding"""
        point_id = self._generate_id()
        point = PointStruct(
            id=point_id, 
            vector=embed.tolist(), 
            payload={"speaker": speaker}
        )
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
        print(f"Upserted embedding with ID {point_id} for speaker '{speaker}'")
        return point_id
    
    def add_speaker_embeddings(self, embeddings: list[np.ndarray], speaker_name: str, metadata: dict[str, any] = None):
        """
        Add multiple embeddings for a single speaker.
        This builds a more robust speaker profile by storing multiple voice samples.
        
        Args:
            embeddings: List of embedding vectors for the speaker
            speaker_name: Name/ID of the speaker
            metadata: Optional additional metadata (e.g., recording date, quality, etc.)
        
        Returns:
            List of inserted point IDs
        """
        if not embeddings:
            print("No embeddings provided")
            return []
        
        points = []
        for idx, embed in enumerate(embeddings):
            payload = {
                "speaker": speaker_name,
                "sample_index": idx,
                "total_samples": len(embeddings)
            }
            
            # Add optional metadata
            if metadata:
                payload.update(metadata)
            
            point = PointStruct(
                id=self._generate_id(),
                vector=embed.tolist(),
                payload=payload
            )
            points.append(point)
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"Added {len(points)} embeddings for speaker '{speaker_name}'")
        return [p.id for p in points]
    
    def get_all_speakers(self):
        """Get list of unique speaker names in the database"""
        # Scroll through all points and extract unique speakers
        speakers = set()
        offset = None
        
        while True:
            result, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False  # Don't need vectors for this
            )
            
            for point in result:
                if point.payload and "speaker" in point.payload:
                    speakers.add(point.payload["speaker"])
            
            if offset is None:
                break
        
        return sorted(list(speakers))
    
    def get_speaker_count(self):
        """Get count of embeddings per speaker"""
        from collections import Counter
        speaker_counts = Counter()
        offset = None
        
        while True:
            result, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            for point in result:
                if point.payload and "speaker" in point.payload:
                    speaker_counts[point.payload["speaker"]] += 1
            
            if offset is None:
                break
        
        return dict(speaker_counts)
    
    def search_similar(self, query_embed, top_k=5, score_threshold=None):
        """Search for similar speaker embeddings"""
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embed.tolist(),
            limit=top_k,
            score_threshold=score_threshold  # Filter by minimum similarity
        )
        return search_result
    
    def identify_speaker(self, query_embed, threshold=0.7):
        """Identify speaker from embedding with confidence threshold"""
        results = self.search_similar(query_embed, top_k=1)
        
        if results and results[0].score >= threshold:
            return {
                "speaker": results[0].payload["speaker"],
                "confidence": results[0].score,
                "match": True
            }
        else:
            return {
                "speaker": None,
                "confidence": results[0].score if results else 0,
                "match": False
            }
        
    def count_embeddings(self):
        """Count total number of embeddings in the collection"""
        stats = self.client.get_collection(collection_name=self.collection_name).points_count
        return stats
    
    def clear_collection(self):
        """Clear all embeddings from the collection"""
        self.client.delete(
            collection_name=self.collection_name,
            filter=None  # Delete all points
        )
        print(f"Cleared all embeddings from collection '{self.collection_name}'")
    
    def delete_collection(self):
        """Delete the entire collection"""
        self.client.delete_collection(collection_name=self.collection_name)
        print(f"Deleted collection '{self.collection_name}'")