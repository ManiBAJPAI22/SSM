# training/evaluate_model.py
import torch
import jiwer
from tqdm import tqdm

def evaluate_model_wer():
    """Evaluate model using Word Error Rate (WER) [82]"""
    model = MultimodalMambaModel()
    model.load_state_dict(torch.load("models/checkpoints/mamba_multimodal.pth"))
    model.eval()
    
    # Load test dataset
    test_dataset = LibriSpeechDataset("data/librispeech_test.pt")
    
    predictions = []
    references = []
    
    with torch.no_grad():
        for item in tqdm(test_dataset):
            audio = item["audio"].unsqueeze(0)
            text_tokens = item["text_tokens"].unsqueeze(0)
            
            # Generate prediction
            logits = model(audio, text_tokens[:, :1])  # Start with first token
            predicted_tokens = torch.argmax(logits, dim=-1)
            
            # Decode to text
            pred_text = decode_tokens(predicted_tokens[0])
            ref_text = item["text"]
            
            predictions.append(pred_text)
            references.append(ref_text)
    
    # Calculate WER
    wer = jiwer.wer(references, predictions)
    print(f"Word Error Rate: {wer:.3f} ({wer*100:.1f}%)")
    
    return wer

if __name__ == "__main__":
    wer = evaluate_model_wer()
