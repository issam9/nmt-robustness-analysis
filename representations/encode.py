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
from utils import tokenize_batch, pool, get_filename, get_punctuated_words, add_special_case


def encode_batch(encoder, tokenized_batch, batch_word_ids, lang_token, pooling):
    word_start_index = 1 if lang_token else 0  # Adjust for language token if present
    
    with torch.no_grad():
        hidden_states = encoder(**tokenized_batch, output_hidden_states=True).hidden_states

    layer_encodings = []
    # Process each layer's hidden states
    for layer_states in hidden_states:
        batch_encodings = []
        for states, word_ids in zip(layer_states, batch_word_ids):
            # Pool embeddings for each word by aggregating the token embeddings corresponding to the word
            sentence_encoding = torch.stack(
                [pool(states[np.where(np.array(word_ids) == i)[0] + word_start_index], pooling) for i in set(word_ids)]
            )
            batch_encodings.append(sentence_encoding.cpu().numpy())  # Move encoding to CPU and store
            assert len(set(word_ids)) == sentence_encoding.shape[0]
        layer_encodings.append(batch_encodings)
    
    word_encodings = np.array(layer_encodings, dtype='object')
    word_encodings = np.transpose(word_encodings, (1, 0))  # Reformat to match expected structure
        
    return word_encodings



# Create a mapping of labels to their corresponding ID
def get_label2id(df):
    label2id = {label: 0 if label == 'clean' else max(df[df['label'] == label].label_ids.tolist()[0])
                for label in df.label.unique()}
    return label2id


def encode(encoder, df, spacy_model, max_length, device, batch_size, lang_token, pooling, encode_clean):
    file_encodings = []
    sequences = df.line.tolist() if encode_clean else df.tokenized_line.tolist()  
    label_ids = df.label_ids.to_numpy()

    # Iterate over the dataset in batches
    for i in range(0, len(df), batch_size):
        batch = sequences[i:i + batch_size]
        batch_label_ids = label_ids[i:i + batch_size]
        tokenized_batch, batch_word_ids = tokenize_batch(tokenizer, spacy_model, batch, device, max_length, lang_token)
        if not all(len(x) == len(set(y)) for x, y in zip(batch_label_ids, batch_word_ids)):
            raise ValueError("Length mismatch")
        batch_encodings = encode_batch(encoder, tokenized_batch, batch_word_ids, lang_token, pooling)
        file_encodings.append(batch_encodings)

    file_encodings = np.concatenate(file_encodings, axis=0)  # Concatenate all batch encodings
    return file_encodings, label_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs="+", default=None, type=str, help='Name of the model on HF')
    parser.add_argument('--data-dir', required=False, default='./data/grammar-noise', type=str)
    parser.add_argument('--output-dir', required=False, default='./outputs/representations/encodings', type=str)
    parser.add_argument('--max-length', required=False, default=1024, type=int, help='Maximum sequence length')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--src-lang', type=str, default='en', help='Source language')
    parser.add_argument('--tgt-lang', type=str, default='es', help='Target language')
    parser.add_argument('--pooling', type=str, default='last', help='Pooling of subword encodings')
    parser.add_argument('--error', type=str, help='Error files to encode')
    parser.add_argument('--encode-clean', action='store_true', help='Encode the clean version of the data')
    
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, f"{args.src_lang}-{args.tgt_lang}")
    
    if args.src_lang == 'en':
        spacy_model = spacy.load("en_core_web_sm")
    elif args.src_lang == 'fr':
        spacy_model = spacy.load("fr_core_news_sm")
    else:
        print(f"Language {args.src_lang} is not supported.")
    
    # Load model configurations from JSON file
    with open('./langcodes.json', 'r') as f:
        langcodes = json.load(f)
    
    models = args.models
    
    for model_name in models:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, early_stopping=False).to(device)
        encoder = model.get_encoder()
        
        model_prefix = model_name.split('/')[-1].split('-')[0].split('_')[0]
        model_langcodes = next((v for k, v in langcodes.items() if model_prefix in k), None)
        print(model_langcodes)

        lang_token = model_langcodes.get(args.src_lang) if model_langcodes else None

        # Ensure max length does not exceed model limits
        max_length = min(args.max_length, model.config.max_position_embeddings)
        
        # Create output directory for saving encodings
        curr_output_dir = os.path.join(output_dir, f"{model_name.split('/')[-1]}/{args.pooling}")
        Path(curr_output_dir).mkdir(exist_ok=True, parents=True)

        error = args.error if args.error else model_prefix
        # Determine files to encode based on error type and clean data flag
        file_patterns = [f"test.{error}.*.pkl"] if args.encode_clean else [
            f"train_probing.{error}.*.pkl",
            f"dev.{error}.*.pkl",
            f"test.{error}.*.pkl"
        ]
        files = [f for pattern in file_patterns for f in glob(os.path.join(args.data_dir, pattern))]
        
        # Process and encode each file
        for file in files:
            print(f"---------- Encoding file {file} using {model_name.split('/')[-1]} ----------")
            save_file = os.path.join(curr_output_dir, get_filename(file))
            df = pd.read_pickle(file)
            punct_words = df['word'].map(lambda x: get_punctuated_words(x) if x is not None else []).tolist()
            punct_words = [item for sublist in punct_words for item in sublist]
            punct_words = set(punct_words)
            add_special_case(spacy_model, punct_words)
            
            df.rename(columns={'label_idx': 'label_ids'}, inplace=True)
            encodings, label_ids = encode(
                encoder, df, spacy_model, max_length, device, args.batch_size, lang_token, args.pooling, args.encode_clean
            )
            label2id = get_label2id(df)

            # Save encoded data and corresponding labels
            file_suffix = ".clean" if args.encode_clean else ""
            np.save(f"{save_file}{file_suffix}.encoding", encodings)
            np.save(f"{save_file}{file_suffix}.label_ids", label_ids)
            np.save(f"{save_file}{file_suffix}.label2id", label2id)

    print("Encoding done")
