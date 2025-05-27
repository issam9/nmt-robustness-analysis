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
from utils import tokenize_batch, pool, get_filename, get_punctuated_words, add_special_case
from modeling import M2M100ForConditionalGeneration, MarianMTModel, MBartForConditionalGeneration


MODEL_MAPPING = {
    'm2m100': M2M100ForConditionalGeneration,
    'nllb': M2M100ForConditionalGeneration,
    'opus': MarianMTModel,
    'mbart': MBartForConditionalGeneration   
}


def encode_batch(encoder, tokenized_batch, batch_word_ids, noisy_word_ids, lang_token, pooling='mean'):
    word_start_index = 0 if lang_token is None else 1
    batch_masked_head_embeddings = []
    num_heads, num_layers = encoder.config.encoder_attention_heads, encoder.config.encoder_layers

    for i in range(num_heads):
        head_mask = torch.ones((num_layers, num_heads)).index_fill(1, torch.tensor([i]), 0.).to(encoder.device)

        with torch.no_grad():
            hidden_states = encoder(**tokenized_batch, head_mask=head_mask, output_hidden_states=True).hidden_states
            layer_word_states = []

            for layer_states in hidden_states:
                batch_states = []
                for states, word_ids, noisy_word_id in zip(layer_states, batch_word_ids, noisy_word_ids):
                    for i in noisy_word_id:
                        token_idxs = np.where(np.array(word_ids) == int(i))[0] + word_start_index
                        word_states = states[token_idxs[0]:token_idxs[-1] + 1, :].mean(0) if pooling == 'mean' else states[token_idxs[-1], :]
                        batch_states.append(word_states.cpu().numpy())

                layer_word_states.append(np.stack(batch_states))
            
            batch_masked_head_embeddings.append(np.stack(layer_word_states, axis=1))
    
    return np.stack(batch_masked_head_embeddings, axis=2)

def encode(encoder, df, spacy_model, max_length, device, batch_size, lang_token, pooling='last'):
    file_encodings = []
    sequences = df.tokenized_line.tolist()
    
    for i in range(0, len(df), batch_size):
        batch = sequences[i:i + batch_size]
        noisy_word_ids = df.loc[i:i + batch_size, 'index'].tolist()
        tokenized_batch, batch_word_ids = tokenize_batch(tokenizer, spacy_model, batch, device, max_length, lang_token)
        batch_encodings = encode_batch(encoder, tokenized_batch, batch_word_ids, noisy_word_ids, lang_token, pooling)
        file_encodings.append(batch_encodings)

    return np.concatenate(file_encodings, axis=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs="+", default=None, type=str, help='Path of the model')
    parser.add_argument('--data-dir', required=False, default='./data/grammar-noise', type=str)
    parser.add_argument('--output-dir', default='outputs/representations/head_masking', type=str)
    parser.add_argument('--max-length', required=True, type=int, help='Maximum sequence length')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--src-lang', type=str, default='en', help='Source language')
    parser.add_argument('--tgt-lang', type=str, default='es', help='Target language')
    parser.add_argument('--error', type=str, help='Error files to encode')
    parser.add_argument('--pooling', type=str, default='mean', help='Pooling strategy')

    args = parser.parse_args()
    

    # data_dir = os.path.join(args.data_dir, f"{args.src_lang}-{args.tgt_lang}")
    output_dir = os.path.join(args.output_dir, f"{args.src_lang}-{args.tgt_lang}")
    
    if args.src_lang == 'en':
        spacy_model = spacy.load("en_core_web_sm")
    elif args.src_lang == 'fr':
        spacy_model = spacy.load("fr_core_news_sm")
    else:
        print(f"Language {args.src_lang} is not supported.")

    with open('./langcodes.json', 'r') as f:
        langcodes = json.load(f)

    print(f"Encoding using the following models: {args.models}")

    for model_name in args.models:
        model_prefix = model_name.split('/')[-1].split('-')[0].split('_')[0]
        CustomAutoModelForSeq2SeqLM = MODEL_MAPPING.get(model_prefix, None)
        assert CustomAutoModelForSeq2SeqLM is not None, f"Model {model_prefix} not supported"
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = CustomAutoModelForSeq2SeqLM.from_pretrained(model_name, early_stopping=False).to(device)
        encoder = model.get_encoder()

        model_langcodes = next((v for k, v in langcodes.items() if model_prefix in k), None)
        print(model_langcodes)

        lang_token = model_langcodes.get(args.src_lang) if model_langcodes else None

        max_length = min(args.max_length, model.config.max_position_embeddings)
        # Create output directory for saving encodings
        curr_output_dir = os.path.join(output_dir, f"{model_name.split('/')[-1]}/{args.pooling}")
        Path(curr_output_dir).mkdir(exist_ok=True, parents=True)

        error = args.error if args.error is not None else model_prefix
        
        files = glob(os.path.join(args.data_dir, f"test.{error}.*.pkl"))
        
        for file in files:
            print(f"---------- Encoding file {file} using {model_name.split('/')[-1]} ----------")
            save_file = os.path.join(curr_output_dir, f"{get_filename(file)}")
            df = pd.read_pickle(file).query("label != 'clean'").reset_index(drop=True)
            punct_words = df['word'].map(lambda x: get_punctuated_words(x) if x is not None else []).tolist()
            punct_words = [item for sublist in punct_words for item in sublist]
            punct_words = set(punct_words)
            add_special_case(spacy_model, punct_words)
            
            df.rename(columns={'label_idx': 'label_ids'}, inplace=True)
            noisy_influences = encode(encoder, df, spacy_model, max_length, device, args.batch_size, lang_token, args.pooling)

            with open(f"{save_file}.masked_head_encodings.pkl", 'wb') as f:
                pickle.dump(noisy_influences, f)

    print("Embeddings saved")
