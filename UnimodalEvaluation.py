from Transformer_torch import Transformer_EEG
from Transformer_torch import Transformer_Vision
from Transformer_torch import Transformer_Audio
import torch
import torch.nn as nn
import pickle
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd


def load_data(modality):

    file_name = f"subject_{sub:02d}_{modality[:3].lower()}.pkl"
    file_ = os.path.join(file_path, modality, file_name)

    with open(file_, 'rb') as f:
        data_list = pickle.load(f)
    [tr_x, tr_y, te_x, te_y] = data_list
    return [tr_x, tr_y, te_x, te_y]


def load_pretrained_model():
    mod_aud_path = os.path.join(os.getcwd(), 'ast-finetuned-audioset')
    Trainer_aud = Transformer_Audio.AudioModelTrainer(data_aud, model_path=mod_aud_path, sub=f"subject_{sub:02d}",
                                                      num_classes=5, weight_decay=1e-5, lr=0.005, batch_size=8)
    Trainer_aud.model.load_state_dict(torch.load(f"ckpt/audio/model_audio_classification_sub{sub:d}.pth", weights_only=True))

    model = Transformer_EEG.EEGClassificationModel(eeg_channel=30)
    Trainer_eeg = Transformer_EEG.EEGModelTrainer(data_eeg, model=model, lr=0.001, batch_size=64)
    Trainer_eeg.model.load_state_dict(torch.load(f"ckpt/eeg/model_eeg_classification_sub{sub:d}.pth", weights_only=True))

    mod_vis_path = os.path.join(os.getcwd(), 'facial_emotions_image_detection')
    Trainer_vis = Transformer_Vision.ImageClassifierTrainer(data_vis, model_path=mod_vis_path, sub=f"subject_{sub:02d}",
                                                            num_labels=5, lr=5e-5, batch_size=128)
    Trainer_vis.model.load_state_dict(torch.load(f"ckpt/vision/model_vision_classification_sub{sub:d}.pth", weights_only=True))

    # return Trainer_aud.model, Trainer_eeg.model, Trainer_vis.model
    return Trainer_aud, Trainer_eeg, Trainer_vis


def hook(module, input, output, layer_name):
    hooked_data[layer_name].append(input[0])
    # return input


# Register hook for each model
def register_hook(model_aud, model_eeg, model_vis):
    model_eeg.model.mlp[0].register_forward_hook(lambda module, input, output: hook(module, input, output, 'eeg_input'))
    model_aud.model.classifier.register_forward_hook(lambda module, input, output: hook(module, input, output, 'aud_input'))
    model_vis.model.classifier.register_forward_hook(lambda module, input, output: hook(module, input, output, 'vis_input'))


def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False


def prepare_dataloader():
    vis_tr = model_vis.preprocess_images(data_vis[0]).view(-1, 3, 224, 224)
    label_tr = torch.from_numpy(np.repeat(model_vis.tr_y, model_vis.frame_per_sample)).long()
    eeg_tr = torch.from_numpy(np.repeat(data_eeg[0], model_vis.frame_per_sample, axis=0)).float()
    aud_tr = model_aud._feature_extract(data_aud[0])
    aud_tr = aud_tr.repeat_interleave(model_vis.frame_per_sample, dim=0)
    train_dataset = TensorDataset(vis_tr, label_tr, eeg_tr, aud_tr)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    vis_te = model_vis.preprocess_images(data_vis[2]).view(-1, 3, 224, 224)
    label_te = torch.from_numpy(np.repeat(model_vis.te_y, model_vis.frame_per_sample)).long()
    eeg_te = torch.from_numpy(np.repeat(data_eeg[2], model_vis.frame_per_sample, axis=0)).float()
    aud_te = model_aud._feature_extract(data_aud[2])
    aud_te = aud_te.repeat_interleave(model_vis.frame_per_sample, dim=0)
    test_dataset = TensorDataset(vis_te, label_te, eeg_te, aud_te)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    return train_loader, test_loader


class MultimodalEmotionRecognitionWithAttention(nn.Module):
    def __init__(self, num_classes, embed_dim=1596, num_heads=4):
        super(MultimodalEmotionRecognitionWithAttention, self).__init__()
        # Self-attention based fusion layer
        self.attention_fusion = SelfAttentionFusion(embed_dim, num_heads)

        # Final classification head
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Optional dropout for regularization

    def forward(self, video_feat, audio_feat, eeg_feat):
        # Concatenate the features from all three modalities
        fused_features = torch.cat((video_feat, audio_feat, eeg_feat), dim=-1)

        # Pass through self-attention based fusion
        fused_attention_output = self.attention_fusion(fused_features)

        # Pass through the classification head
        x = self.relu(self.fc1(fused_attention_output))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SelfAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super(SelfAttentionFusion, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.fc = nn.Linear(embed_dim, 512)  # Dimensionality reduction

    def forward(self, fused_features):
        # Self-attention expects the input in (sequence_length, batch_size, embed_dim)
        fused_features = fused_features.unsqueeze(0)  # Add sequence length dimension
        attention_output, _ = self.self_attention(fused_features, fused_features, fused_features)
        attention_output = attention_output.squeeze(0)  # Remove sequence length dimension
        return self.fc(attention_output)  # Pass through fully connected layer



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for sub in range(1, 43):
        print("subject", sub)
        file_path = "/home/yk/EAV_dataset/Input_images/"
        data_eeg, data_aud, data_vis = load_data('EEG'), load_data('Audio'), load_data('Vision')
        model_aud, model_eeg, model_vis = load_pretrained_model()
        register_hook(model_aud, model_eeg, model_vis)
        _, testloader = prepare_dataloader()
        model_aud.model.to(device), model_eeg.model.to(device), model_vis.model.to(device)
        model = MultimodalEmotionRecognitionWithAttention(num_classes=5)
        state_dict = torch.load(f"ckpt/multimodal/model_subject_{sub:02d}.pth", weights_only=True)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.to(device)

        model_aud.model.eval()
        model_eeg.model.eval()
        model_vis.model.eval()
        model.eval()

        test_correct = 0
        test_total = 0
        test = {
            'vis_correct': 0,
            'aud_correct': 0,
            'eeg_correct': 0,
            'multimodal_correct': 0,
            'total': 0
        }

        with torch.no_grad():  # Disable gradient computation for evaluation
            for batch_idx, (b_vis, b_lbl, b_eeg, b_aud) in enumerate(testloader, start=1):
                b_vis, b_lbl, b_eeg, b_aud = [b.to(device) for b in (b_vis, b_lbl, b_eeg, b_aud)]
                hooked_data = {
                    'eeg_input': [],
                    'aud_input': [],
                    'vis_input': []
                }
                o_vis = model_vis.model(b_vis)
                o_eeg = model_eeg.model(b_eeg)
                o_aud = model_aud.model(b_aud)

                _, predicted_vis = torch.max(o_vis.logits, 1)
                _, predicted_eeg = torch.max(o_eeg, 1)
                _, predicted_aud = torch.max(o_aud.logits, 1)
                test['vis_correct'] += (predicted_vis == b_lbl).sum().item()
                test['aud_correct'] += (predicted_aud == b_lbl).sum().item()
                test['eeg_correct'] += (predicted_eeg == b_lbl).sum().item()

                vis_input = torch.cat([x.to(device) for x in hooked_data['vis_input']], dim=0)
                aud_input = torch.cat([x.to(device) for x in hooked_data['aud_input']], dim=0)
                eeg_input = torch.cat([x.to(device) for x in hooked_data['eeg_input']], dim=0)
                output = model(vis_input, aud_input, eeg_input)
                _, predicted = torch.max(output, 1)
                test['multimodal_correct'] += (predicted == b_lbl).sum().item()
                test['total'] += b_lbl.size(0)

        # Calculate accuracies
        vis_accuracy = test['vis_correct'] / test['total'] * 100
        aud_accuracy = test['aud_correct'] / test['total'] * 100
        eeg_accuracy = test['eeg_correct'] / test['total'] * 100
        multimodal_accuracy = test['multimodal_correct'] / test['total'] * 100

        print(f"Vision Accuracy: {vis_accuracy:.2f}%")
        print(f"Audio Accuracy: {aud_accuracy:.2f}%")
        print(f"EEG Accuracy: {eeg_accuracy:.2f}%")
        print(f"Multimodal Accuracy: {multimodal_accuracy:.2f}%")

        # Append the results for the current subject
        results.append({
            'Subject': sub,
            'Vision Accuracy (%)': vis_accuracy,
            'Audio Accuracy (%)': aud_accuracy,
            'EEG Accuracy (%)': eeg_accuracy,
            'Multimodal Accuracy (%)': multimodal_accuracy
        })
    # Create a DataFrame and save it to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('model_test_accuracies.csv', index=False)
