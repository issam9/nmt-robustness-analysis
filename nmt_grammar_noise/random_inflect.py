from sacremoses import MosesDetokenizer
import random, lemminflect
from typing import Dict
from morpheus_en_fr.utils import map_tag, get_forme, get_fr_lemmas, get_forme_multi, get_punctuated_words
import numpy as np
import argparse
import pandas as pd
import codecs
import tqdm
from collections import Counter
from inflecteur import inflecteur
from spacy_lefff import POSTagger
from spacy.language import Language
import spacy
from spacy.attrs import ORTH

np.random.seed(123)

'''
inflection_counts should have the following structure:
{ PTB tag: int, ... , PTB tag: int }
'''

def random_inflect(source: str, inflected_counts: Dict[str, int], inflection_counts: Dict[str,int]=None, lang: str='en') -> str:
    have_inflections = {'NOUN', 'VERB', 'ADJ'}
    universal_to_infelect = {'NOUN': 'Nom', 'VERB': 'Verbe', 'ADJ': 'Adjectif'}
    doc = nlp(source)
    tokenized = [token.text for token in doc]
    spaces = [token.whitespace_ for token in doc]
    tags = [token.tag_ if lang=='en' else token._.melt_tagger for token in doc]
    upper = False
    if tokenized[0][0].isupper():
        upper = True
        tokenized[0]= tokenized[0].lower()
    
    pos_tagged = [map_tag(lang, tag) for tag in tags]       # map to universal
    
    inflected_dist = [inflected_counts.get(tag, 0) for tag in tags]
    if sum(inflected_dist) == 0:
        print('No inflections found', source, tags)
        return tokenized, source, None, None, None, [0] * len(tokenized), 'clean'
    inflected_dist = np.array(inflected_dist) / sum(inflected_dist)
    inflected_size = sum(np.array(inflected_dist) > 0)
    inflected_idx = np.random.choice(range(len(tokenized)), p=inflected_dist, size=inflected_size, replace=False)
    words = [(i.item(), tokenized[i]) for i in inflected_idx]
    
    for i, word in words:
        lemmas = lemminflect.getAllLemmas(word) if lang=='en' else get_fr_lemmas(word, inflecteur)
        inflection = None
        lemma = None
        if lemmas and pos_tagged[i] in have_inflections:
            word_tag = pos_tagged[i] if lang=='en' else universal_to_infelect.get(pos_tagged[i], 'X')
            if word_tag not in lemmas:
                continue
            lemma = lemmas[word_tag][0]
            if lang=='en':
                inflections = [(tag, infl)
                                    for tag, tup in 
                                    lemminflect.getAllInflections(lemma, upos=pos_tagged[i]).items() 
                                    for infl in tup if infl != word]
            else:   # fr
                df = inflecteur.dico_transformer.loc[(inflecteur.dico_transformer.lemma == lemma) & (inflecteur.dico_transformer.index!=word)].reset_index(drop=False)
                inflections = [(tag, infl) for tup, infl in zip(df.forme, df.part) for tag in tup.split(":")]
                
            if inflections:
                if inflection_counts:
                    counts = [inflection_counts[tag] for tag, infl in inflections]
                    counts_dist = np.array(counts) / sum(counts)
                    
                    inflection = np.random.choice([item[1] for item in inflections], p=counts_dist).item()
                else:
                    inflection = np.random.choice([item[1] for item in inflections], size=1).item()
                tokenized[i] = inflection
        
        if inflection is not None:
            break
        
    if upper:
        tokenized[0] = tokenized[0].title()
    
    doc = spacy.tokens.Doc(nlp.vocab, words=tokenized, spaces=spaces)
    label_ids = [0] * len(tokenized)
    label_ids[i] = 1 if inflection is not None else 0
    label = 'noisy' if inflection is not None else 'clean'
    return tokenized, doc.text, i, inflection, lemma, label_ids, label


def random_inflect_multi(source: str, inflected_counts: Dict[str, int], inflection_counts: Dict[str,int]=None, lang: str='en') -> str:
    have_inflections = {'NOUN', 'VERB', 'ADJ'}
    universal_to_infelect = {'NOUN': 'Nom', 'VERB': 'Verbe', 'ADJ': 'Adjectif'}
    doc = nlp(source)
    tokenized = [token.text for token in doc]
    spaces = [token.whitespace_ for token in doc]
    tags = [token.tag_ if lang=='en' else token._.melt_tagger for token in doc]
    upper = False
    if tokenized[0][0].isupper():
        upper = True
        tokenized[0]= tokenized[0].lower()
    
    pos_tagged = [map_tag(lang, tag) for tag in tags]       # map to universal
    
    inflected_dist = [inflected_counts.get(tag, 0) for tag in tags]
    if sum(inflected_dist) == 0:
        print('No inflections found', source, tags)
        return tokenized, source, None, None, None, [0] * len(tokenized), 'clean'
    inflected_dist = np.array(inflected_dist) / sum(inflected_dist)
    inflected_size = sum(np.array(inflected_dist) > 0)
    inflected_idx = np.random.choice(range(len(tokenized)), p=inflected_dist, size=inflected_size, replace=False)
    words = [(i.item(), tokenized[i]) for i in inflected_idx]
    
    final_infl = []
    final_infl_idx = []
    final_lemma = []
    for i, word in words:
        lemmas = lemminflect.getAllLemmas(word) if lang=='en' else get_fr_lemmas(word, inflecteur)
        inflection = None
        lemma = None
        if lemmas and pos_tagged[i] in have_inflections:
            word_tag = pos_tagged[i] if lang=='en' else universal_to_infelect.get(pos_tagged[i], 'X')
            if word_tag not in lemmas:
                continue
            lemma = lemmas[word_tag][0]
            if lang=='en':
                inflections = [(tag, infl)
                                    for tag, tup in 
                                    lemminflect.getAllInflections(lemma, upos=pos_tagged[i]).items() 
                                    for infl in tup]
            else:   # fr
                df = inflecteur.dico_transformer.loc[(inflecteur.dico_transformer.lemma == lemma)].reset_index(drop=False)
                inflections = [(tag, infl) for tup, infl in zip(df.forme, df.part) for tag in tup.split(":")]
                
            if inflections:
                if inflection_counts:
                    counts = [inflection_counts[tag] for tag, infl in inflections]
                    counts_dist = np.array(counts) / sum(counts)
                    
                    inflection = np.random.choice([item[1] for item in inflections], p=counts_dist).item()
                else:
                    inflection = np.random.choice([item[1] for item in inflections], size=1).item()
                if tokenized[i] != inflection:
                    tokenized[i] = inflection
                    final_infl.append(inflection)
                    final_infl_idx.append(i)
                    final_lemma.append(lemma)
    if upper:
        tokenized[0] = tokenized[0].title()
    
    doc = spacy.tokens.Doc(nlp.vocab, words=tokenized, spaces=spaces)
    label_ids = [0] * len(tokenized)
    for i in final_infl_idx:
        label_ids[i] = 1
    label = 'noisy' if inflection else 'clean'
    return tokenized, doc.text, final_infl_idx, final_infl, final_lemma, label_ids, label

def add_special_case(nlp, cases):
    for case in cases:
        special_case = [{ORTH: case}] 
        nlp.tokenizer.add_special_case(case, special_case)
        
def main_multi(args):

    counts_df = pd.read_pickle(args.counts_file)
    punct_words = counts_df['word'].map(lambda x: get_punctuated_words(x) if x is not None else []).tolist()
    punct_words = [item for sublist in punct_words for item in sublist]
    punct_words = set(punct_words)
    add_special_case(nlp, punct_words)
    
    detokenizer = MosesDetokenizer(lang=args.src_lang)
    
    counts_df['detokenized_line'] = counts_df['tokenized_line'].apply(lambda x: detokenizer.detokenize(x))
    
    def spacy_tag_multi(x, column='detokenized_line', lang='en'):
        doc = nlp(x[column])
        if lang=='en':
            tags = [token.tag_ for token in doc]
        elif lang=='fr':
            tags = [token._.melt_tagger for token in doc]
        else:
            raise ValueError("Language not supported")
        return [tags[i] for i in x['index']]
    
    counts_df['inflected_pos'] = counts_df.apply(lambda x: spacy_tag_multi(x, 'line', args.src_lang) if x['word'] is not None else None, axis=1)

    inflected_dict = counts_df.inflected_pos.explode().value_counts().to_dict()
    
    if args.src_lang == 'en':
        counts_df['inflection_pos'] = counts_df.apply(lambda x: spacy_tag_multi(x) if x['word'] is not None else None, axis=1)

        inflection_dict = counts_df.inflection_pos.explode().value_counts().to_dict()
    elif args.src_lang == 'fr':
        counts_df['forme'] = counts_df.apply(lambda x: get_forme_multi(x, inflecteur) if x['word'] is not None else None, axis=1)
        inflection_dict = Counter([f for item in counts_df['forme'].tolist() if item is not None for f in set(item.split(":"))])
    else:
        raise ValueError("Language not supported")

    with codecs.open(args.source_file, 'r', 'utf-8') as f:
        sentences = [sentence.strip() for sentence in f.readlines()]
    
    perturbed = []
    detokenized_perturbed = []
    inflection_idx = []
    inflection = []
    lemma = []
    label_ids = []
    labels = []
    for source in tqdm.tqdm(sentences):
        curr_perturbed, curr_detokenized_perturbed, curr_token, curr_infl, curr_lemma, curr_label_ids, curr_label = \
            random_inflect_multi(source, inflected_dict, inflection_dict, lang=args.src_lang)
        
        perturbed.append(curr_perturbed)
        detokenized_perturbed.append(curr_detokenized_perturbed)
        inflection_idx.append(curr_token)
        inflection.append(curr_infl)
        lemma.append(curr_lemma)
        label_ids.append(curr_label_ids)
        labels.append(curr_label)

    output = dict(line=sentences, tokenized_line=perturbed, detokenized_line=detokenized_perturbed, index=inflection_idx, 
                  word=inflection, lemma=lemma, label_ids=label_ids, label=labels)
    
    df = pd.DataFrame.from_dict(output)
    df.to_pickle(args.output_file)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Random inflection')
    parser.add_argument('--src-lang', type=str, default='en', help='source language')
    parser.add_argument('--source-file', type=str, help='input file')
    parser.add_argument('--output-file', type=str, help='output file')
    parser.add_argument('--counts-file', type=str, help='inflection counts file')
    args = parser.parse_args()
    
    if args.src_lang=='en':
        print("Loading English model")
        nlp = spacy.load("en_core_web_sm")
    else:
        nlp = spacy.load("fr_core_news_sm")
        Language.factory("french_pos", func=lambda nlp, name: POSTagger())
        nlp.add_pipe("french_pos", name='pos_lefff')

        inflecteur = inflecteur()
        inflecteur.load_dict()
        
    # main_multi(args)
    

    counts_df = pd.read_pickle(args.counts_file)

    detokenizer = MosesDetokenizer(lang=args.src_lang)
    
    # if 'detokenized_line' not in counts_df.columns:
    counts_df['detokenized_line'] = counts_df['tokenized_line'].apply(lambda x: detokenizer.detokenize(x))

    def spacy_tag(x, column='detokenized_line', lang='en'):
        doc = nlp(x[column])
        if lang=='en':
            tags = [token.tag_ for token in doc]
        elif lang=='fr':
            tags = [token._.melt_tagger for token in doc]
        else:
            raise ValueError("Language not supported")
        assert len(tags) == len(x['tokenized_line']), f"Length mismatch: {len(tags)} != {len(x['tokenized_line'])}, {x[column]}"
        return tags[int(x['index'][0])]
    
    counts_df['inflected_pos'] = counts_df.apply(lambda x: spacy_tag(x, 'line', args.src_lang) if x['word'] is not None else None, axis=1)

    inflected_dict = counts_df.inflected_pos.value_counts().to_dict()
    
    if args.src_lang == 'en':
        counts_df['inflection_pos'] = counts_df.apply(lambda x: spacy_tag(x) if x['word'] is not None else None, axis=1)

        inflection_dict = counts_df.inflection_pos.value_counts().to_dict()
    elif args.src_lang == 'fr':
        counts_df['forme'] = counts_df.apply(lambda x: get_forme(x, inflecteur) if x['word'] is not None else None, axis=1)
        inflection_dict = Counter([f for item in counts_df['forme'].tolist() if item is not None for f in set(item.split(":"))])
    else:
        raise ValueError("Language not supported")

    with codecs.open(args.source_file, 'r', 'utf-8') as f:
        sentences = [sentence.strip() for sentence in f.readlines()]
    
    # random_indices = np.random.choice(len(sentences), 100, replace=False)
    # sentences = [sentences[i] for i in random_indices]
    
    perturbed = []
    detokenized_perturbed = []
    inflection_idx = []
    inflection = []
    lemma = []
    label_ids = []
    labels = []
    preds = []
    queries = []
    for source in tqdm.tqdm(sentences):
        curr_perturbed, curr_detokenized_perturbed, curr_token, curr_infl, curr_lemma, curr_label_ids, curr_label = \
            random_inflect(source, inflected_dict, inflection_dict, lang=args.src_lang)
        
        perturbed.append(curr_perturbed)
        detokenized_perturbed.append(curr_detokenized_perturbed)
        inflection_idx.append(curr_token)
        inflection.append(curr_infl)
        lemma.append(curr_lemma)
        label_ids.append(curr_label_ids)
        labels.append(curr_label)

    output = dict(line=sentences, tokenized_line=perturbed, detokenized_line=detokenized_perturbed, index=inflection_idx, 
                  word=inflection, lemma=lemma, label_ids=label_ids, label=labels)
    
    df = pd.DataFrame.from_dict(output)
    df.to_pickle(args.output_file)
    