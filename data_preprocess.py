import pandas as pd
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import emoji
from tqdm import tqdm
from torch.utils.data import Dataset
from openai import OpenAI
import time



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_the_model_gpt2(model_name):
    
    model_name = "gpt2"  # "gpt2-medium" or "gpt2-large" for bigger models
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, padding_side='left')
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token


    model.eval()
    model.to(device)

    return model, tokenizer


def paraphrase_gpt2_batch(texts, model, tokenizer, num_return_sequences=3, gpt_batch_size=64):
    all_paraphrases = []

    print("[INFO] Splitting GPT-2 work into small batches...")

    for i in tqdm(range(0, len(texts), gpt_batch_size)):
        batch_texts = texts[i:i + gpt_batch_size]
        prompts = [f"Paraphrase this sentence:\n{text}\nParaphrase:" for text in batch_texts]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

        # Dynamically calculate max_length based on the longest input + 20
        input_lengths = (inputs['attention_mask'] > 0).sum(dim=1)
        max_input_len = input_lengths.max().item()
        max_length = min(max_input_len + 20, tokenizer.model_max_length)  # cap to model max

        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )

        for k in range(len(batch_texts)):
            paraphrases = []
            for j in range(num_return_sequences):
                idx = k * num_return_sequences + j
                decoded = tokenizer.decode(outputs[idx], skip_special_tokens=True)
                clean = decoded.split("Paraphrase:")[-1].strip().replace('\\', '')
                paraphrases.append(clean)
            all_paraphrases.append(paraphrases)

    return all_paraphrases



# Using chat gpt model
from config import api_key
client = OpenAI(api_key=api_key)  # Your API key

def emoji_to_text_translation(texts, gpt_batch_size=64):

    """
    Utilizing ChatGPT-o4-mini model to translate the emoji
    
    """
    translations = []

    print("[INFO] Translating emoji to text using chatgpt-o4 mini...")

    for i in tqdm(range(0, len(texts), gpt_batch_size)):
        batch_texts = texts[i:i + gpt_batch_size]

        for text in batch_texts:
            prompt = (
                "Rewrite the following sentence by replacing each emoji with an appropriate English word or short phrase that describes the emotion or meaning of the emoji. "
                "Keep the sentence as close to the original as possible, but make sure the meaning of the emoji is clearly expressed in words. Do not just remove the emoji—translate its feeling or context.\n"
                f"Sentence: {text}\nRewritten:"
            )
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a careful translator who turns emojis into clear English."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=128,
                    temperature=0.3,
                )
                output = response.choices[0].message.content.strip()
                if output.lower().startswith("rewritten:"):
                    output = output[len("rewritten:"):].strip()
                translations.append(output)

            except Exception as e:
                translations.append(f"[ERROR: {e}]")
                time.sleep(1)
    return translations



def replace_emojis_with_words(text):
    # Replace emojis with CLDR short names (e.g., ":smiling_face:")
    text_with_names = emoji.demojize(text)
    text_cleaned = text_with_names.replace(":", "").replace("_", " ")
    
    return text_cleaned


def get_text():

    path_pos = 'dataset/1k_data_tweets_emoticon_pos.csv'
    path_neg = 'dataset/1k_data_tweets_emoticon_neg.csv'
    path_posneg = 'dataset/1k_data_emoji_tweets_senti_posneg.csv'

    df_pos = pd.read_csv(path_pos)
    df_neg = pd.read_csv(path_neg)
    pos_neg = pd.read_csv(path_posneg)

    data_df = pd.concat([df_pos, df_neg], ignore_index=True)
    data_df = pd.concat([data_df, pos_neg], ignore_index=True)

    X_text = data_df['post'].astype(str).tolist()

    return X_text



def get_dataset(texts, dynamic_score=3, model_name='gpt2'):

    emoji_list = []
    new_list = []

    # Pre-clean all texts
    cleaned_sentences = [replace_emojis_with_words(s) for s in texts]

    
    if model_name == 'gpt2':
        model, tokenizer = set_the_model_gpt2(model_name='gpt2')
        all_rewritten = paraphrase_gpt2_batch(cleaned_sentences, model, tokenizer, num_return_sequences=dynamic_score)

        for orig_sentence, paraphrases in zip(texts, all_rewritten):
            for rewritten in paraphrases:
                emoji_list.append(orig_sentence)
                new_list.append(rewritten)

    if model_name == 'chatgpt':

        all_rewritten = emoji_to_text_translation(cleaned_sentences)
        
        for orig_sentence, paraphrases in zip(texts, all_rewritten):
            emoji_list.append(orig_sentence)
            new_list.append(paraphrases)



    return emoji_list, new_list



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





if __name__ == '__main__':
    # for testing only
    all_texts = get_text()
    input_texts, target_texts = get_dataset(all_texts[:100], model_name='chatgpt')

    for i in target_texts:
        print(i)