import pandas as pd
import numpy as np
import os
from pathlib import Path
import pickle
import cka_lib 


# Utility function to compute CKA distance
def compute_cka_distance(influence_states, original_states):
    output = np.empty(influence_states.shape[:-1])
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            for k in range(output.shape[2]):
                output[i][j][k] = 1 - cka_lib.cka(influence_states[i][j][k], original_states[i][j+1])
    return output


def get_original_word_states(states, df):
    indices = df[df['label'].isin(['clean'])].index
    df = df[~df['label'].isin(['clean'])]
    df = df.reset_index(drop=True)
    states = np.delete(states, indices, axis=0)
    indices = [[int(i) for i in idx] for idx in df['index']]
    num_samples = len([i for idx in indices for i in idx])
    num_layers = len(states[0])
    dim = len(states[0][0][0])
    output = np.empty((num_samples, num_layers, dim))
    c = 0
    for i in range(len(states)):
        for k in indices[i]:
            for j in range(num_layers):
                output[c][j] = states[i][j][k]
            c += 1
    return output



# Function to compute and save the CKA distances
def compute_and_save_distances(src, lang, model_prefix, finetuned_model, base_model, error, output_folder, to_clean=False):
    output_folder = f"{output_folder}/{src}-{lang}"
    influence_file = lambda model, err, base: f'{output_folder}/{model}{"" if base else "-"+err}/last/test.{err if err!="morpheus" else model_prefix+".morpheus"}.{src}.masked_head_encodings.pkl'
    if to_clean:
        encoding_file = lambda model, err, base: (
            f'./outputs/representations/encodings/{src}-{lang}/{model}'
            f'{"" if base else "-"+err}/last/test.{err if err!="morpheus" else model_prefix+".morpheus"}.{src}.clean.encoding.npy'
        )
    else:
        encoding_file = lambda model, err, base: (
            f'./outputs/representations/encodings/{src}-{lang}/{model}'
            f'{"" if base else "-"+err}/last/test.{err if err!="morpheus" else model_prefix+".morpheus"}.{src}.encoding.npy'
        )
    df_file = f'./data/data/grammar-noise/{src}-{lang}/test.{error if error!="morpheus" else model_prefix+".morpheus"}.{src}.pkl'

    def load_pickle(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def load_npy(filepath):
        return np.load(filepath, allow_pickle=True)

    def save_pickle(data, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    # Load the required files
    df = pd.read_pickle(df_file)
    
    # Noisy output
    print(f"---- Computing word influence for {finetuned_model}-{error} ----")
    influence_states = load_pickle(influence_file(finetuned_model, error, False))
    original_states = load_npy(encoding_file(finetuned_model, error, False))
    original_word_states = get_original_word_states(original_states, df)
    noisy_output = compute_cka_distance(influence_states, original_word_states)

    # Clean output
    print(f"---- Computing word influence for {finetuned_model}-clean-{error} ----")
    influence_states = load_pickle(influence_file(finetuned_model, error, False))
    original_states = load_npy(encoding_file(f"{finetuned_model}-clean", error, False))
    original_word_states = get_original_word_states(original_states, df)
    clean_output = compute_cka_distance(influence_states, original_word_states)

    # Base output
    print(f"---- Computing word influence for {base_model} ----")
    influence_states = load_pickle(influence_file(base_model, error, True))
    original_states = load_npy(encoding_file(base_model, error, True))
    original_word_states = get_original_word_states(original_states, df)
    base_output = compute_cka_distance(influence_states, original_word_states)


    # Save the outputs
    if to_clean:
        save_pickle(noisy_output, f'{output_folder}/{finetuned_model}-{error}/last/test.{error}.{src}.to_clean.distance.pkl')
        save_pickle(clean_output, f'{output_folder}/{finetuned_model}-clean-{error}/last/test.{error}.{src}.to_clean.distance.pkl')
        save_pickle(base_output, f'{output_folder}/{base_model}/last/test.{error}.{src}.to_clean.distance.pkl')
    else:
        save_pickle(noisy_output, f'{output_folder}/{finetuned_model}-{error}/last/test.{error}.{src}.distance.pkl')
        save_pickle(clean_output, f'{output_folder}/{finetuned_model}-clean-{error}/last/test.{error}.{src}.distance.pkl')
        save_pickle(base_output, f'{output_folder}/{base_model}/last/test.{error}.{src}.distance.pkl')


# Main processing loop
if __name__ == "__main__":
    errors = ['prep', 'nounnum', 'article', 'morpheus']
    languages = ["es", "de", "it", "nl"]
    sources = ["en"]
    output_folder = './outputs/representations/head_masking'

    models = {
        'opus': 'opus-mt',
        'm2m100': 'm2m100_418M',
        'mbart': 'mbart-large-50-many-to-many-mmt',
        'nllb': 'nllb-200-distilled-600M'
    }

    for lang in languages:
        # We only have results for Fr-Es
        for src in sources:
            if src == 'fr' and lang != 'es':
                continue
            for model_prefix, base_model_name in models.items():
                finetuned_model = f'{model_prefix}-{src}-{lang}'
                if 'opus' in model_prefix:
                    base_model_name = f'{base_model_name}-{src}-{lang}'

                for error in errors:
                    compute_and_save_distances(src, lang, model_prefix, finetuned_model, base_model_name, error, output_folder)
                    compute_and_save_distances(src, lang, model_prefix, finetuned_model, base_model_name, error, output_folder, to_clean=True)