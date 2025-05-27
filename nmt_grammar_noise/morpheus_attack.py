from morpheus_en_fr.morpheus_nmt import MorpheusHuggingfaceNMT
import argparse
import codecs
import pandas as pd
import numpy as np
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Morpheus attack')
    parser.add_argument('--src-lang', type=str, default='en', help='source language')
    parser.add_argument('--tgt-lang', type=str, default=None, help='target language')
    parser.add_argument('--model', type=str, default='Helsinki-NLP/opus-mt-en-es', help='model name')
    parser.add_argument('--source-file', type=str, help='input file')
    parser.add_argument('--target-file', type=str, help='input file')
    parser.add_argument('--output-file', type=str, help='output file')
    parser.add_argument('--multilingual', action='store_true', help='is multilingual model')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    parser.add_argument('--multi', action='store_true', help='multiple errors per sentence')
    args = parser.parse_args()
    
    morpheus_nmt = MorpheusHuggingfaceNMT(args.model, src_lang=args.src_lang, tgt_lang=args.tgt_lang, multilingual=args.multilingual, batch_size=args.batch_size)

    with codecs.open(args.source_file, 'r', 'utf-8') as f:
        sentences = [sentence.strip() for sentence in f.readlines()]

    with codecs.open(args.target_file, 'r', 'utf-8') as f:
        references = [reference.strip() for reference in f.readlines()]
    
    perturbed, detokenized_perturbed, inflection_idx, inflection, lemma, label_ids, labels, preds, queries = morpheus_nmt.morph_sentences(sentences, references, constrain_pos=True, multi=args.multi)
    
    output = dict(line=sentences, tokenized_line=perturbed, detokenized_line=detokenized_perturbed, index=inflection_idx, 
                  word=inflection, lemma=lemma, label_idx=label_ids, label=labels, preds=preds, queries=queries)
    
    new_df = pd.DataFrame.from_dict(output)
    
    df = new_df
    df.to_pickle(args.output_file)
    