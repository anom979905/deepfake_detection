import os
import torch
import torch.nn as nn
from torchvision import models
from django.conf import settings

class DeepfakeDetectionModel(nn.Module):
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(DeepfakeDetectionModel, self).__init__()
        base_model = models.resnext50_32x4d(pretrained=True)
        # Use all layers except the last two (typical feature extractor)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional=bidirectional)
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(hidden_dim if bidirectional else 2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)  # flatten batch and sequence dims
        features = self.feature_extractor(x)
        x = self.avgpool(features)
        x = x.view(batch_size, seq_len, -1)  # reshape for LSTM
        lstm_out, _ = self.lstm(x)
        x = self.dropout(lstm_out[:, -1, :])  # take last output in sequence
        out = self.linear(x)
        return features, out

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(sequence_length, num_classes=2):
    device = get_device()

    # This function picks the best model file based on sequence length
    models_folder = os.path.join(settings.PROJECT_DIR, "models")
    model_files = [f for f in os.listdir(models_folder) if f.endswith("pt")]

    # Filter models matching the sequence length (assuming filename pattern contains it)
    matching_models = []
    for f in model_files:
        parts = f.split("_")
        try:
            seq = int(parts[3])
            if seq == sequence_length:
                matching_models.append(f)
        except (IndexError, ValueError):
            pass

    if not matching_models:
        raise FileNotFoundError(f"No model found for sequence length {sequence_length}")

    # Select model with highest accuracy (assuming accuracy is second part of filename)
    def get_accuracy(filename):
        try:
            return float(filename.split("_")[1])
        except:
            return 0.0

    best_model_file = max(matching_models, key=get_accuracy)
    model_path = os.path.join(models_folder, best_model_file)

    # Initialize model and load weights
    model = DeepfakeDetectionModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model
