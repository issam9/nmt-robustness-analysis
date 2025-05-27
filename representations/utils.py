from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os
from spacy.attrs import ORTH
import string



# Helper function to extract filename without extension
def get_filename(path):
    return os.path.splitext(os.path.basename(path))[0]

# Tokenize sentence and return word ids mapping for each token
def tokenize(tokenizer, spacy_model, words, max_length, lang_token):
    # If input is not a list, tokenize it using the spacy model
    if not isinstance(words, list):
        words = [token.text for token in spacy_model(words.strip())]
    
    word_ids = [] 
    token_ids = []
    num_special_toks = 2 if lang_token else 1  # Account for special tokens
    
    # Loop over each word to get token ids and assign a word index
    for i, word in enumerate(words):
        ids = tokenizer.encode(word, add_special_tokens=False)
        word_ids.extend([i] * len(ids))  # Track which tokens belong to which word
        token_ids.extend(ids)
        
        # Ensure tokenized sequence doesn't exceed max length
        if len(token_ids) > max_length - num_special_toks:
            token_ids = token_ids[:max_length - num_special_toks]
            word_ids = word_ids[:max_length - num_special_toks]
            break

    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id # Determine end of sequence token
    if lang_token:
        lang_token_id = tokenizer.encode(lang_token, add_special_tokens=False)
        token_ids = lang_token_id + token_ids + [eos_token_id]  # Add language token and EOS
    else:
        token_ids.append(eos_token_id)  # Append EOS token if no language token
    
    # Ensure the word and token length matches (except for special tokens)
    assert len(word_ids) == len(token_ids) - num_special_toks
    return token_ids, word_ids


# Tokenize a batch of sequences
def tokenize_batch(tokenizer, spacy_model, batch, device, max_length, lang_token):
    tokenized_batch = {"input_ids": []}
    batch_word_ids = []
    
    # Tokenize each sequence in the batch
    for input_ids, word_ids in map(lambda x: tokenize(tokenizer, spacy_model, x, max_length, lang_token), batch):
        tokenized_batch['input_ids'].append(input_ids)
        batch_word_ids.append(word_ids)

    tokenized_batch = tokenizer.pad(tokenized_batch)  # Pad sequences to the same length
    tokenized_batch = {k: torch.tensor(v).to(device) for k, v in tokenized_batch.items()}  # Move tensors to device
    return tokenized_batch, batch_word_ids


# Pooling function to reduce the token embeddings for each word
def pool(embeddings, pooling='mean'):
    if pooling == 'mean':
        return torch.mean(embeddings, dim=0)  # Average embeddings for mean pooling
    elif pooling == 'first':
        return embeddings[0]  # Use the first token's embedding
    return embeddings[-1]  # Use the last token's embedding (default)



def get_punctuated_words(words):
    punct_words = []
    for word in words:
        """Checks if a word contains any punctuation characters."""
        for char in word:
            if char in string.punctuation:
                punct_words.append(word)
    return punct_words


def add_special_case(nlp, cases):
    for case in cases:
        special_case = [{ORTH: case}] 
        nlp.tokenizer.add_special_case(case, special_case)