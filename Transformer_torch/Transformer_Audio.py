import torch
import torch.nn as nn
import torch.optim as optim
# from transformers import AutoModelForAudioClassification
from transformers_local import ASTForAudioClassification as AutoModelForAudioClassification
from torch.utils.data import DataLoader, TensorDataset
from transformers import ASTFeatureExtractor
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
from layers_ours import *
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class AudioModelTrainer:
    def __init__(self, DATA, model_path, sub = '', num_classes=5, weight_decay=1e-5, lr=0.001, batch_size=64):

        self.tr, self.tr_y, self.te, self.te_y = DATA
        self.tr_x = self._feature_extract(self.tr)
        self.te_x = self._feature_extract(self.te)

        self.sub = sub
        self.batch_size = batch_size

        self.train_dataloader = self._prepare_dataloader(self.tr_x, self.tr_y, shuffle=True)
        self.test_dataloader = self._prepare_dataloader(self.te_x, self.te_y, shuffle=False)

        self.model = AutoModelForAudioClassification.from_pretrained(model_path)
        # Modify classifier to fit the number of classes
        # self.model.classifier.dense = torch.nn.Linear(self.model.classifier.dense.in_features, num_classes)
        self.model.classifier = Linear(self.model.config.hidden_size, num_classes)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Setup optimizer and loss function
        self.initial_lr = lr
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.initial_lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def _prepare_dataloader(self, x, y, shuffle=False):
        dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return dataloader

    def _feature_extract(self, x):
        feature_extractor = ASTFeatureExtractor()
        ft = feature_extractor(x, sampling_rate=16000, padding='max_length',
                               return_tensors='pt')
        # spectrogram = ft['input_values'][0].numpy()
        # spectrogram = np.abs(spectrogram.T)
        # # power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
        #
        # plt.figure(figsize=(10, 6))
        # librosa.display.specshow(spectrogram, x_axis='time', y_axis='mel', cmap='magma', hop_length=160)
        # plt.colorbar(label='Intensity')
        # plt.title('Audio Spectrogram')
        # plt.show()

        return ft['input_values']

    def train(self, epochs=20, lr=None, freeze=True):
        lr = lr if lr is not None else self.initial_lr
        if lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        if isinstance(self.model, nn.DataParallel):
            self.model = self.model.module
        # Freeze or unfreeze model parameters based on the freeze flag
        for param in self.model.parameters():
            param.requires_grad = not freeze
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        # Wrap the model with DataParallel
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        for epoch in range(epochs):
            self.model.train()
            train_correct, train_total = 0, 0

            total_batches = len(self.train_dataloader)
            for batch_idx, batch in enumerate(self.train_dataloader, start=1):
                #print(f'batch ({batch_idx}/{total_batches})')

                x, t = [b.to(self.device) for b in batch]
                self.optimizer.zero_grad()
                logits = self.model(x).logits
                loss = self.loss_fn(logits, t)
                if loss.dim() > 0:
                    loss = loss.mean()
                else:
                    loss = loss
                loss.backward()
                self.optimizer.step()

                train_correct += (logits.argmax(dim=-1) == t).sum().item()
                train_total += t.size(0)
            train_accuracy = train_correct / train_total

            self.model.eval()
            correct, total = 0, 0
            outputs_batch = []
            with torch.no_grad():
                for x, t in self.test_dataloader:
                    x, t = x.to(self.device), t.long().to(self.device)
                    logits = self.model(x).logits
                    correct += (logits.argmax(dim=-1) == t).sum().item()
                    total += t.size(0)

                    logits_cpu = logits.detach().cpu().numpy()
                    outputs_batch.append(logits_cpu)
                test_accuracy = correct / total
            if epoch == epochs-1 and not freeze: # we saved test prediction only at last epoch, and finetuning
                self.outputs_test = np.concatenate(outputs_batch, axis=0)

            print(f"Epoch {epoch + 1}/{epochs}, Training Accuracy: {train_accuracy * 100:.2f}%, Test Accuracy: {test_accuracy * 100:.2f}%")
            with open('training_performance_audio.txt', 'a') as f:
                f.write(f"{self.sub}, Epoch {epoch + 1}, Test Accuracy: {test_accuracy * 100:.2f}%\n")

    def save_model(self, sub):
        # Save the trained model for the current subject
        model_path = "ckpt/audio_new"
        os.makedirs(model_path, exist_ok=True)
        model_save_path = f'{model_path}/model_audio_classification_sub{sub:02d}.pth'
        # torch.save(self.model.state_dict(), model_save_path)
        if torch.cuda.device_count() > 1:
            torch.save(self.model.module.state_dict(), model_save_path)
        else:
            torch.save(self.model.state_dict(), model_save_path)
        print(f"Model for Subject {sub} saved as {model_save_path}")


if __name__ == '__main__':
    import pickle
    import os
    test_acc = []
    for idx in range(11, 43):

        file_name = f"subject_{idx:02d}_aud.pkl"
        # file_ = os.path.join(os.getcwd(), 'Feature_vision', file_name)

        file_ = os.path.join(os.getcwd(), '../EAV_dataset/Input_images/Audio', file_name)

        with open(file_, 'rb') as f:
            vis_list2 = pickle.load(f)
        tr_x_vis, tr_y_vis, te_x_vis, te_y_vis = vis_list2

        mod_path = os.path.join(os.getcwd(), 'ast-finetuned-audioset')
        data = [tr_x_vis, tr_y_vis, te_x_vis, te_y_vis]
        trainer = AudioModelTrainer(data,
                                         model_path=mod_path, sub = f"subject_{idx:02d}",
                                         num_classes=5, lr=5e-5, batch_size=32)

        trainer.train(epochs=20, lr=5e-4, freeze=True)
        trainer.train(epochs=10, lr=5e-6, freeze=False)
        trainer.save_model(idx)

        test_acc.append(trainer.outputs_test)

    # import pickle
    # with open("test_acc_vision.pkl", 'wb') as f:
    #     pickle.dump(test_acc, f)
    #
    #
    # # test accuracy for 200 trials
    # ## acquire the test label from one subject, it is same for all subjects
    # from sklearn.metrics import f1_score
    # file_name = f"subject_{1:02d}_vis.pkl"
    # # file_ = os.path.join(os.getcwd(), 'Feature_vision', file_name)
    # file_ = os.path.join(os.getcwd(), '../EAV_dataset/Input_images/Vision', file_name)
    #
    # with open(file_, 'rb') as f:
    #     vis_list2 = pickle.load(f)
    # te_y_vis = vis_list2[3]
    #
    # # load test accuracy for all subjects: 5000 (200, 25) predictions
    # with open("test_acc_vision.pkl", 'rb') as f:
    #     testacc = pickle.load(f)
    #
    # test_acc_all = list()
    # test_f1_all = list()
    # for sub in range(42):
    #     aa = testacc[sub]
    #     aa2 = np.reshape(aa, (200, 25, 5), 'C')
    #     aa3 = np.mean(aa2, 1)
    #     out1 = np.argmax(aa3, axis = 1)
    #     accuracy = np.mean(out1 == te_y_vis)
    #     test_acc_all.append(accuracy)
    #
    #     f1 = f1_score(te_y_vis, out1, average='weighted')
    #     test_f1_all.append(f1)
    #
    # test_acc_all = np.reshape(np.array(test_acc_all), (42, 1))
    # test_f1_all = np.reshape(np.array(test_f1_all), (42, 1))