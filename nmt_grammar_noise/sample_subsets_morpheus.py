import pandas as pd
import glob
from pathlib import Path
import shutil
import argparse
from sacremoses import MosesDetokenizer
import os


def process_files(src_lang, tgt_langs, data_path, seed=123):
    for lang in tgt_langs:
        # Get all files matching the error type for the language pair
        files = glob.glob(f'{data_path}/{src_lang}-{lang}/morpheus/train.*')
        print(files)
        print(os.getcwd())
        save_path = f'data/finetuning-subset/morpheus/{src_lang}/{lang}'
        
        for split in ['train', 'dev', 'test']:
            Path(save_path + f"/{split}").mkdir(exist_ok=True, parents=True)
        
        for file in files:
            df = pd.read_pickle(file)
            
            # Sample 30% of the data grouped by error label
            df_sample = df.groupby('label', as_index=False).apply(lambda x: x.sample(frac=0.3, random_state=seed))
            indices = df_sample.droplevel(0).index 
            
            # Load and drop the sampled sentences from source language file
            with open(f'./data/europarl-st/{src_lang}/{lang}/train/segments.{src_lang}', 'r') as f:
                source_sentences = f.readlines()
            
            source_sentences = pd.Series(source_sentences).drop(indices)
            
            # Save the cleaned source sentences
            with open(Path(save_path).joinpath(f'train/segments.clean.{file.split(".")[-4]}.{src_lang}'), 'w') as f:
                for sentence in source_sentences:
                    f.write(sentence)
            
            # Load and drop the sampled sentences from target language file
            with open(f'./data/europarl-st/{src_lang}/{lang}/train/segments.{lang}', 'r') as f:
                target_sentences = f.readlines()
            
            target_sentences = pd.Series(target_sentences).drop(indices)
            
            # Save the cleaned target sentences
            with open(Path(save_path).joinpath(f'train/segments.clean.{file.split(".")[-4]}.{lang}'), 'w') as f:
                for sentence in target_sentences:
                    f.write(sentence)
            
            # Update the DataFrame by dropping the sampled rows and saving the new files
            df = df.drop(indices).reset_index(drop=True)
            df_sample = df_sample.reset_index(drop=True)
            print(file)
            df_sample.to_pickle(file.replace('train', 'train_probing'))  # Save probing set
            df.to_pickle(file.replace('train', 'train_finetuning'))  # Save finetuning set
        
    # Copy test and dev splits
    for split in ['test', 'dev']:
        shutil.copy(f'./data/europarl-st/{src_lang}/{lang}/{split}/segments.{src_lang}', Path(save_path).joinpath(f'{split}/segments.{src_lang}'))
        shutil.copy(f'./data/europarl-st/{src_lang}/{lang}/{split}/segments.{lang}', Path(save_path).joinpath(f'{split}/segments.{lang}'))



# Save source and target sentences as text files for finetuning and evaluation
def pkl_to_txt(src_lang, tgt_langs):
    detokenizer = MosesDetokenizer(lang=src_lang)
    for lang in tgt_langs:
        for split in ['train_finetuning', 'dev', 'test']:
            for model in ['opus', 'm2m100']:
                save_path = f'./data/finetuning-subset/morpheus/{src_lang}/{lang}/{split.split("_")[0]}'
                Path(save_path).mkdir(exist_ok=True, parents=True)
                # Load the pickle file for the given split and error type
                pkl_file_path = f'./data/grammar-noise/{src_lang}-{lang}/morpheus/{split}.{model}.{src_lang}.pkl'
                df = pd.read_pickle(pkl_file_path)

                # Detokenize the sentences
                sentences = df.tokenized_line.tolist()
                detokenized_sentences = [detokenizer.detokenize(sentence) for sentence in sentences]

                # Save detokenized sentences to the segment file
                segment_file_path = Path(save_path).joinpath(f'segments.{model}.{src_lang}')
                with open(segment_file_path, 'w') as f:
                    for sentence in detokenized_sentences:
                        f.write(sentence + '\n')
                
                print(f"{lang}---{split}----{len(sentences)} sentences saved to {segment_file_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process files for language pairs and split PKL files into segments.")
    
    parser.add_argument('--src-lang', type=str, required=True, help="Source language")
    parser.add_argument('--tgt-langs', type=str, nargs='+', required=True, help="Target languages")
    parser.add_argument('--data-path', type=str, default='./data/grammar-noise', help="Path to data directory")
    parser.add_argument('--seed', type=int, default=123, help="Random seed for sampling")
    
    args = parser.parse_args()
        
    process_files(args.src_lang, args.tgt_langs, args.data_path, args.seed)

    pkl_to_txt(args.src_lang, args.tgt_langs)
