import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset
from emoji_map import emoji_mapping, emoji_descriptions




def insert_emoji_randomly(text, label):
    """
    Insert the emoji back into tests on random location
    """
    emoji = emoji_mapping.get(label, "")
    words = text.split()
    
    if not words:
        return emoji  # Just return emoji if sentence is empty

    insert_pos = random.randint(0, len(words))  # could be at the start or end
    words.insert(insert_pos, emoji)

    return " ".join(words)


def insert_emoji_create_input_data(text_data, label_data, iteration=10):
    """
    Create the input dataset
    """
    input_dataset = []
    input_dataset_label = []
    for _ in range(iteration):
        emoji_texts = [insert_emoji_randomly(t, l) for t, l in zip(text_data, label_data)]
        input_dataset.extend(emoji_texts)
        input_dataset_label.extend(label_data)

    return input_dataset, input_dataset_label


def replace_emoji_with_description(emoji_text, label):
    
    emoji = emoji_mapping[label]
    description = emoji_descriptions[label]
    return emoji_text.replace(emoji, description)


def get_dataset(texts, labels):
    """
    Returns:
        -- texts data with emoji inserted
        -- pure texts data with emoji translated
    """


    input_emoji_texts, all_labels_extended = insert_emoji_create_input_data(
        text_data=texts,
        label_data=labels
    )

    input_pure_texts = [replace_emoji_with_description(t, l) for t, l in zip(input_emoji_texts, all_labels_extended)]

    return input_emoji_texts, input_pure_texts



class EmojiToTextDataset(Dataset):
    """
    Get the dataset ready for LM
    """
    def __init__(self, input_texts, target_texts, tokenizer, max_input_len=128, max_target_len=64):
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        source = self.tokenizer(
            self.input_texts[idx],
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        target = self.tokenizer(
            self.target_texts[idx],
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": source["input_ids"].squeeze(),
            "attention_mask": source["attention_mask"].squeeze(),
            "labels": target["input_ids"].squeeze()
        }
