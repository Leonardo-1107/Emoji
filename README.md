# Emojibag 😄🤖📝: Multimodal Emoji Understanding for Sentiment Classification


## 🎯 Goal of the project
The goal of this project is to create a model that can explain the meaning of emojis in different textual contexts, like the word-bag model. Emojis can convey very different emotions depending on the surrounding text. For example:
	•	Ahh, I see 😄 ...   → shows happiness or relief
	•	Ahh, I see 😭 that's fine   → shows disappointment or sadness
	•	Ahh, I see 😓 call you later   → shows worry or stress

By building a model that translates emojis according to their unique context, we aim to make machines better at truly understanding the intent and emotions behind digital conversations.

---
## 🚀 Overview

This project explores the impact of **explicit emoji-to-text translation** on sentiment classification tasks using social media data. By leveraging advanced language models (GPT2 and ChatGPT) to interpret emojis, we generate paired datasets for training a T5 model (Emojibag) that can better understand and explain the meaning of emojis in various contexts. The trained model is then evaluated in downstream sentiment analysis experiments, demonstrating the benefits of **multimodal representation** in NLP.  
💬 ➡️ 🤗 ➡️ 🎯

---

## 📁 Code Structure

| File               | Purpose                                                                 |
|--------------------|-------------------------------------------------------------------------|
| `config.py`        | 🔑 API key configuration                                                |
| `data_preprocess.py` | 🤖 Emoji translation with GPT2 & ChatGPT; create training data for T5  |
| `train.py`         | 🏋️ Train the Emojibag T5 model for emoji interpretation                 |
| `utils.py`         | 🛠️ Utility functions: data tools, training plots, evaluation, etc.      |

---

## 🛠️ How to Use

1. **Set Up Your API Key**  
   Add your OpenAI API key in `config.py`. 🔑


2. **Train the Model**  
   Use `train.py` to fine-tune the Emojibag T5 model on your dataset. 🏋️

   ```python
    # with GPT2 as emoji-to-text data processor
    python train.py --emoji_translate_model gpt2 --model t5-small

    # with ChatGPT-4o-mini as emoji-to-text data processor
    python train.py --emoji_translate_model gpt2 --model t5-small
   ```

3. **Evaluate Performance**  
   Use `experiments.ipynb` for visualizations 📊 and running downstream sentiment classification experiments. 🎯

---

## ⚙️ Trained Models

The trained model files can be found in [here](https://drive.google.com/drive/folders/1gZ0sEO5osw7fqLdQBNxyzgitpiTL_EOg?usp=sharing)
