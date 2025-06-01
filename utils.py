import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class BertVectorizer:
    def __init__(self, model, tokenizer, device='cuda', batch_size=32, max_length=128):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

    def _encode(self, texts):
        self.model.eval()
        all_embeddings = []

        texts = list(texts)
        dataloader = DataLoader(texts, batch_size=self.batch_size)
        for batch in tqdm(dataloader, desc='Encoding with BERT'):
            encoded = self.tokenizer(list(batch), return_tensors='pt', padding=True, truncation=True, max_length=self.max_length).to(self.device)
            with torch.no_grad():
                outputs = self.model(**encoded)
            last_hidden = outputs.last_hidden_state
            mask = encoded['attention_mask'].unsqueeze(-1)
            pooled = (last_hidden * mask).sum(1) / mask.sum(1)
            all_embeddings.append(pooled.cpu())

        return torch.cat(all_embeddings).numpy()

    def fit_transform(self, texts):
        return self._encode(texts)

    def transform(self, texts):
        return self._encode(texts)
    
    

def plot_train(loss_record, epochs, save_name):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), loss_record, marker='*', linestyle='-', color='b', label='Training Loss')
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_name}.png')