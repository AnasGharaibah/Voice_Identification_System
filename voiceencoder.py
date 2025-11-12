# import necessary libraries
from resemblyzer import preprocess_wav, VoiceEncoder
from  pathlib import Path 
from tqdm import tqdm 
import numpy as np 
from IPython.display import Audio, display
from itertools import groupby
import heapq
import sounddevice as sd
import soundfile as sf
import torch

class VoiceEncoderModule:
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = VoiceEncoder(self.device)

    def preprocess_audio(self, audio_path):
        return preprocess_wav(audio_path)

    def embed_utterance(self, wav):
        return self.encoder.embed_utterance(wav)

    def embed_from_path(self, audio_path):
        wav = self.preprocess_audio(audio_path)
        return self.embed_utterance(wav)
    
    def embed_batch(self, audio_dir):
        """
        Process all wav files in a directory and return embeddings
        
        Args:
            audio_dir: Path to directory containing .wav files
            
        Returns:
            embeddings: numpy array of embeddings
            speakers: list of speaker names (file stems)
        """
        wav_fpaths = list(Path(audio_dir).glob("*.wav"))
        
        if not wav_fpaths:
            raise ValueError(f"No .wav files found in {audio_dir}")
        
        speakers = [wav_fpath.stem for wav_fpath in wav_fpaths]
        embeddings = []
        
        for wav_fpath in wav_fpaths:
            wav = self.preprocess_audio(wav_fpath)
            embedding = self.embed_utterance(wav)
            embeddings.append(embedding)
        
        return np.array(embeddings), speakers




# Example usage:
def main():
    voice_encoder = VoiceEncoderModule()

    # Preprocess and embed a single audio file
    audio_path = "sounds_in"
    #wav = voice_encoder.preprocess_audio(audio_path)
    embedding,speakers = voice_encoder.embed_batch(audio_path)
    
    print("Embedding shape:", embedding , "speakers:", speakers)    


if __name__ == "__main__":
    main()