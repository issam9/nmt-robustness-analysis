from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import numpy as np
import pandas as pd
from glob import glob
import os
from pathlib import Path
import argparse
import json
import spacy
import pickle
from utils import tokenize_batch, get_filename


# Encoding function to compute word attention weights
def encode_batch(encoder, tokenized_batch, batch_word_ids, noisy_word_ids, lang_token):
    word_start_index = 1 if lang_token else 0
    with torch.no_grad():
        attentions = encoder(**tokenized_batch, output_attentions=True).attentions
        attentions = torch.stack(attentions, dim=1)  # Combine layers

    noisy_word_attentions = []
    for layer_attention, word_ids, noisy_word_id in zip(attentions, batch_word_ids, noisy_word_ids):
        layer_attentions = []
        for attention in layer_attention:
            token_idxs = np.where(np.array(word_ids) == noisy_word_id)[0] + word_start_index
            word_attention = attention[:, token_idxs[0]:token_idxs[-1] + 1, :].mean(-2)  # Mean pooling across subwords
            
            # Adjust based on lang_token presence
            if lang_token:
                word_attention = torch.stack([word_attention[:, 0]] + 
                                             [word_attention[:, np.where(np.array(word_ids) == i)[0] + word_start_index].sum(dim=-1) 
                                              for i in set(word_ids)] + 
                                             [word_attention[:, -1]], dim=-1)
            else:
                word_attention = torch.stack([word_attention[:, np.where(np.array(word_ids) == i)[0] + word_start_index].sum(dim=-1)
                                              for i in set(word_ids)] + 
                                             [word_attention[:, -1]], dim=-1)
            
            layer_attentions.append(word_attention.cpu().numpy())
        
        noisy_word_attentions.append(layer_attentions)

    return noisy_word_attentions


def encode(encoder, df, spacy_model, max_length, device, batch_size, lang_token, clean_word_attentions):
    encoder_attentions = []
    sequences = df.line.tolist() if clean_word_attentions else df.tokenized_line.tolist()

    for i in range(0, len(df), batch_size):
        batch = sequences[i:i+batch_size]
        noisy_word_ids = df.loc[i:i+batch_size, 'index'].map(int).tolist()
        tokenized_batch, batch_word_ids = tokenize_batch(tokenizer, spacy_model, batch, device, max_length, lang_token)
        batch_attentions = encode_batch(encoder, tokenized_batch, batch_word_ids, noisy_word_ids, lang_token)
        encoder_attentions.extend(batch_attentions)

    return encoder_attentions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs="+", type=str, help='Model names from Hugging Face', required=True)
    parser.add_argument('--data-dir', required=True, type=str)
    parser.add_argument('--output-dir', type=str, default='outputs/representations/attention_weights')
    parser.add_argument('--max-length', required=True, type=int, help='Maximum sequence length')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--src-lang', type=str, default='en', help='Source language')
    parser.add_argument('--tgt-lang', type=str, default='es', help='Target language')
    parser.add_argument('--error', type=str, required=False, help='Error files to encode')

    args = parser.parse_args()

    if args.src_lang == 'en':
        spacy_model = spacy.load("en_core_web_sm")
    elif args.src_lang == 'fr':
        spacy_model = spacy.load("fr_core_news_sm")
    else:
        print(f"Language {args.src_lang} is not supported.")

    data_dir = os.path.join(args.data_dir, f"{args.src_lang}-{args.tgt_lang}")
    
    # Load model configs
    with open('./configs.json', 'r') as f:
        configs = json.load(f)

    for model_name in args.models:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        encoder = model.get_encoder()

        # Check if the model uses multilingual tokens
        langcodes = configs['multilingual'].get(model_name.split('/')[-1].split('-')[0].split('_')[0], {})
        lang_token = langcodes.get(args.src_lang, None)

        max_length = min(args.max_length, model.config.max_position_embeddings)
        output_dir = os.path.join(args.output_dir, f"{os.path.basename(data_dir)}/{model_name.split('/')[-1]}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Load and process files
        files = glob(os.path.join(data_dir, f"test.{args.error}*.pkl"))
        for file in files:
            df = pd.read_pickle(file)
            # We ignore clean instances
            df = df[df['label'] != 'clean'].reset_index(drop=True)

            print(f"Processing {file} with {model_name}...")

            noisy_attentions = encode(encoder, df, spacy_model, max_length, device, args.batch_size, lang_token, clean_word_attentions=False)
            clean_attentions = encode(encoder, df, spacy_model, max_length, device, args.batch_size, lang_token, clean_word_attentions=True)

            save_file = os.path.join(output_dir, f"{get_filename(file)}")
            with open(save_file + ".clean_word.attention_weights.pkl", 'wb') as f:
                pickle.dump(clean_attentions, f)
            with open(save_file + ".attention_weights.pkl", 'wb') as f:
                pickle.dump(noisy_attentions, f)

    print("Attention weights saved.")
