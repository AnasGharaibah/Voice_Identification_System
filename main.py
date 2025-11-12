from database import VectorDatabase
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

from voiceencoder import VoiceEncoderModule



qdrant_client = QdrantClient(host="localhost", port=6333)

#upload embeddings to Qdrant
voiceencoder = VoiceEncoderModule()

embedding, speakers = voiceencoder.embed_batch("speaker_sounds")

db = VectorDatabase(client=qdrant_client, collection_name="Voice_Embeddings_1")

db.upsert_embeddings(embedding, speakers)
print('='*70)

embedding_1 = voiceencoder.embed_from_path("testing/Mandelas Sound Bites [TubeRipper.cc].wav")
result = db.search_similar(embedding_1)

result_2  = db.get_all_speakers()

for res in result:
    print(f"Speaker: {res.payload['speaker']}, Score: {res.score}")

print('='*70)
print("All speakers in database:")
print(result_2)
