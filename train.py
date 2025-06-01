import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


from data_preprocess import get_text, get_dataset, EmojiToTextDataset
from utils import plot_train


import argparse
def get_args():
    parser = argparse.ArgumentParser(description="Train a T5 model with custom settings.")
    
    parser.add_argument("--emoji_translate_model", type=str, default='gpt2', help="Model applied to translate the emoji from the input texts")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--model", type=str, default="t5-small", help="Emojibag Model name or path")

    return parser.parse_args()


if __name__ == "__main__":
    
    args = get_args()

    print("========== Parsed Arguments ==========")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("======================================\n\n")


    BATCH_SIZE = args.batch_size
    LR = args.lr
    EPOCHS = args.epochs
    MODEL = args.model
    Translate_Emoji_Model = args.emoji_translate_model


    # get the data ready
    all_texts = get_text()
    input_texts, target_texts = get_dataset(all_texts, model_name=Translate_Emoji_Model)

    print(f"[INFO] Total length of input emojibag data {len(input_texts)}\n\n\n[INFO] Training starts ...")


    # prepare the model and train
    model = T5ForConditionalGeneration.from_pretrained(MODEL)
    tokenizer = T5Tokenizer.from_pretrained(MODEL, legacy=False)
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
    for epoch in range(EPOCHS):
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
                torch.save(model.state_dict(), f'{MODEL}_{EPOCHS}_{Translate_Emoji_Model}.pt')

        
    plot_train(loss_record, EPOCHS, save_name=f'{MODEL}_{EPOCHS}_{Translate_Emoji_Model}')


