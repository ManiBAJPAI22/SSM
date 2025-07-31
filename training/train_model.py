# training/train_model.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import jiwer  # For WER calculation

class LibriSpeechDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "audio": torch.tensor(item["audio_features"], dtype=torch.float32),
            "text_tokens": torch.tensor(item["text_tokens"], dtype=torch.long),
            "text": item["text"]
        }

def calculate_wer(predictions, references):
    """Calculate Word Error Rate [82]"""
    return jiwer.wer(references, predictions)

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalMambaModel().to(device)
    
    # Load data
    dataset = LibriSpeechDataset("data/librispeech_processed.pt")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter("runs/mamba_training")
    
    model.train()
    for epoch in range(5):  # Small number for MVP
        total_loss = 0
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            audio = batch["audio"].to(device)
            text_tokens = batch["text_tokens"].to(device)
            
            # Forward pass
            logits = model(audio, text_tokens[:, :-1])  # Teacher forcing
            targets = text_tokens[:, 1:].reshape(-1)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                writer.add_scalar("Loss/Train", loss.item(), epoch * len(dataloader) + batch_idx)
        
        print(f"Epoch {epoch}: Average Loss = {total_loss / len(dataloader):.4f}")
    
    # Save model
    torch.save(model.state_dict(), "models/checkpoints/mamba_multimodal.pth")
    return model

if __name__ == "__main__":
    trained_model = train_model()
