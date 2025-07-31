# training/prepare_data.py
from datasets import load_dataset
import librosa
import numpy as np
import torch
from transformers import AutoTokenizer

def prepare_librispeech_data():
    # Load LibriSpeech train-clean-100 (118 hours, manageable size)
    dataset = load_dataset("librispeech_asr", "clean", split="train.100")
    
    # Initialize tokenizer for text processing
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    processed_data = []
    for i, sample in enumerate(dataset):
        if i >= 1000:  # Limit for MVP (adjust based on compute)
            break
            
        # Process audio: extract 80-dim log-mel spectrograms
        audio = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        
        # Resample to 16kHz if needed
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=16000, n_mels=80, n_fft=1024, hop_length=256
        )
        log_mel = np.log(mel_spec + 1e-8)
        
        # Tokenize text
        text_tokens = tokenizer.encode(sample["text"], max_length=512, truncation=True)
        
        processed_data.append({
            "audio_features": log_mel,
            "text_tokens": text_tokens,
            "text": sample["text"]
        })
    
    return processed_data

# Run data preparation
processed_data = prepare_librispeech_data()
torch.save(processed_data, "data/librispeech_processed.pt")
