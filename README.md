# Voice Identification System with Qdrant Vector Database

A speaker identification system that uses voice embeddings and vector similarity search to identify speakers in audio files.

## Overview

This project implements a voice identification system that:
- Extracts voice embeddings from audio files using Resemblyzer
- Stores speaker embeddings in a Qdrant vector database
- Identifies speakers by comparing query audio against stored voice profiles

## Features

- **Voice Embedding Extraction**: Convert audio files into high-dimensional speaker embeddings
- **Vector Database Storage**: Store and manage speaker profiles using Qdrant
- **Speaker Identification**: Match unknown voices against a database of known speakers
- **Batch Processing**: Process multiple audio files efficiently

## Prerequisites

- Python 3.7+
- Docker (for running Qdrant)
- Audio files in `.wav` format

## Installation

1. **Install Python dependencies:**
```bash
pip install qdrant-client resemblyzer numpy sounddevice soundfile torch
```

2. **Start Qdrant vector database:**
```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 qdrant/qdrant
```

Access the Qdrant dashboard at: `http://localhost:6333/dashboard`

## Project Structure

```
├── database.py          # Qdrant vector database operations
├── voiceencoder.py      # Voice embedding extraction module
├── main.py              # Example: Upload and search speaker embeddings
├── speaker_sounds/      # Directory for reference speaker audio files
└── testing/             # Directory for test audio files
```

## Usage

### 1. Building a Speaker Database

Store reference audio samples for known speakers:

```python
from database import VectorDatabase
from voiceencoder import VoiceEncoderModule
from qdrant_client import QdrantClient

# Initialize components
qdrant_client = QdrantClient(host="localhost", port=6333)
voiceencoder = VoiceEncoderModule()
db = VectorDatabase(client=qdrant_client, collection_name="Voice_Embeddings_1")

# Extract embeddings from audio files
embedding, speakers = voiceencoder.embed_batch("speaker_sounds")

# Upload to database
db.upsert_embeddings(embedding, speakers)
```

### 2. Identifying a Speaker

Match an unknown voice against your database:

```python
# Extract embedding from test audio
query_embedding = voiceencoder.embed_from_path("testing/unknown_speaker.wav")

# Search for similar speakers
results = db.search_similar(query_embedding, top_k=5)

for res in results:
    print(f"Speaker: {res.payload['speaker']}, Confidence: {res.score:.2f}")
```

### 3. Managing Speaker Database

```python
# Get all registered speakers
all_speakers = db.get_all_speakers()
print("Registered speakers:", all_speakers)

# Get embedding count per speaker
speaker_counts = db.get_speaker_count()
print("Embeddings per speaker:", speaker_counts)

# Identify speaker with confidence threshold
result = db.identify_speaker(query_embedding, threshold=0.7)
if result['match']:
    print(f"Identified: {result['speaker']} (confidence: {result['confidence']:.2f})")
else:
    print("Unknown speaker or low confidence match")
```

## Key Components

### `VectorDatabase` (database.py)
Manages speaker embeddings in Qdrant:
- `upsert_embeddings()`: Store multiple speaker embeddings
- `upsert_one_embedding()`: Store a single embedding
- `add_speaker_embeddings()`: Store multiple samples for one speaker
- `search_similar()`: Find similar voices with cosine similarity
- `identify_speaker()`: Identify speaker with confidence threshold
- `get_all_speakers()`: List all registered speakers
- `get_speaker_count()`: Count embeddings per speaker
- `count_embeddings()`: Get total number of embeddings
- `clear_collection()`: Remove all embeddings
- `delete_collection()`: Delete the entire collection

### `VoiceEncoderModule` (voiceencoder.py)
Handles audio processing and embedding extraction:
- `preprocess_audio()`: Prepare audio for encoding
- `embed_utterance()`: Generate embedding from preprocessed audio
- `embed_from_path()`: Process and embed audio file in one step
- `embed_batch()`: Process multiple audio files from a directory

## Configuration

### Vector Database Settings
- **Vector Size**: 256 dimensions (default)
- **Distance Metric**: Cosine similarity
- **Collection Name**: Customizable per use case

### Identification Thresholds
- **Default threshold**: 0.7 (configurable)
- **Confident match**: Score > 0.75
- **Uncertain match**: Score > 0.65
- **Unknown speaker**: Score ≤ 0.65

## Example Workflow

The `main.py` script demonstrates a complete workflow:

1. **Initialize the system:**
   - Connect to Qdrant database
   - Create voice encoder instance
   - Set up vector database collection

2. **Upload reference speakers:**
   - Process all WAV files in `speaker_sounds/` directory
   - Extract voice embeddings
   - Store in Qdrant with speaker labels

3. **Test identification:**
   - Load a test audio file
   - Extract its embedding
   - Search for similar speakers in database
   - Display results with confidence scores

4. **View database contents:**
   - List all registered speakers
   - Check embedding counts

## Advanced Features

### Multiple Samples Per Speaker

For more robust speaker profiles, add multiple voice samples:

```python
# Collect multiple samples for one speaker
sample_embeddings = [
    voiceencoder.embed_from_path("speaker1_sample1.wav"),
    voiceencoder.embed_from_path("speaker1_sample2.wav"),
    voiceencoder.embed_from_path("speaker1_sample3.wav")
]

# Add all samples with metadata
db.add_speaker_embeddings(
    embeddings=sample_embeddings,
    speaker_name="John Doe",
    metadata={"recording_date": "2025-11-12", "quality": "high"}
)
```

### Custom Similarity Search

```python
# Search with custom parameters
results = db.search_similar(
    query_embed=query_embedding,
    top_k=10,  # Return top 10 matches
    score_threshold=0.8  # Only return scores above 0.8
)
```

## Notes

- Audio files should be in WAV format for best results
- Longer reference audio samples (5-10 seconds) improve accuracy
- Multiple samples per speaker create more robust profiles
- GPU acceleration available for faster processing (set `device="cuda"`)
- Ensure Qdrant is running before executing the scripts

## Troubleshooting

- **Connection Error**: Verify Qdrant is running on `localhost:6333`
- **No Audio Files Found**: Check that WAV files exist in the specified directory
- **Low Confidence Scores**: Try using longer or clearer audio samples
- **Memory Issues**: Process audio files in smaller batches

## License

This project uses the Resemblyzer library for voice encoding. Please refer to the Resemblyzer license for usage terms.
