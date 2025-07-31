# training/export_onnx.py
import torch
import torch.onnx
from multimodal_mamba import MultimodalMambaModel

def export_to_onnx():
    model = MultimodalMambaModel()
    model.load_state_dict(torch.load("models/checkpoints/mamba_multimodal.pth"))
    model.eval()
    
    # Create dummy inputs
    dummy_audio = torch.randn(1, 80, 100)  # [batch, mel_bins, time]
    dummy_text = torch.randint(0, 50257, (1, 50))  # [batch, seq_len]
    
    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_audio, dummy_text),
        "models/onnx/mamba_multimodal.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['audio_input', 'text_input'],
        output_names=['output'],
        dynamic_axes={
            'audio_input': {2: 'audio_time'},
            'text_input': {1: 'text_length'},
            'output': {1: 'sequence_length'}
        }
    )
    print("Model exported to ONNX successfully!")

if __name__ == "__main__":
    export_to_onnx()
