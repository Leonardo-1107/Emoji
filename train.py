import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


from data_preprocess import get_dataset, EmojiToTextDataset
from utils import plot_train


import argparse
def get_args():
    parser = argparse.ArgumentParser(description="Train a T5 model with custom settings.")
    
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--model", type=str, default="t5-small", help="Model name or path")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    BATCH_SIZE = args.batch_size
    LR = args.lr
    EPOCHS = args.epochs
    MODEL = args.model

    # get the data ready
    train = pd.read_parquet('emojibag_data/train-00000-of-00001.parquet')
    test = pd.read_parquet('emojibag_data/test-00000-of-00001.parquet')
    all_data = pd.concat([train, test])
    label_select = [1, 2, 5, 6, 9, 14, 16, 19]
    filtered_data = all_data[all_data["label"].isin(label_select)].reset_index(drop=True)

    all_texts = filtered_data['text'].to_list()
    all_labels = filtered_data['label'].to_list()
    input_texts, target_texts = get_dataset(all_texts, all_labels)

    # prepare the model and train
    model = T5ForConditionalGeneration.from_pretrained(MODEL)
    tokenizer = T5Tokenizer.from_pretrained(MODEL)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    dataset = EmojiToTextDataset(
        input_texts,
        target_texts,
        tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    best_loss = None
    loss_record = []
    for epoch in tqdm(range(EPOCHS), desc='Training Process'):
        model.train()
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"[INFO] Epoch {epoch+1} loss: {loss.item():.4f}")
        loss_record.append(loss.item())

        if best_loss is None:
            best_loss = loss.item()
        else:
            if loss.item() < best_loss:
                print("[INFO] Saving the best model as .pt  ...")
                best_loss = loss.item()
                torch.save(model.state_dict(), f'{MODEL}_{EPOCHS}.pt')

        
    plot_train(loss_record, EPOCHS, save_name=f'{MODEL}_{EPOCHS}')


