from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
import argparse
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import evaluate
import json
import comet
import pandas as pd
import os
from pathlib import Path
import spacy


# Load evaluation metrics
metric_bleu = evaluate.load("sacrebleu")
metric_meteor = evaluate.load("meteor")
metric_chrf = evaluate.load("chrf")
metric_comet = comet.load_from_checkpoint(comet.download_model("Unbabel/wmt22-comet-da"))

# Dataset class to handle list-based datasets
class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]

# Post-process predictions and labels
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def save_metrics(target_file, metrics):
    with open(target_file, 'w') as f:
        json.dump(metrics, f, indent=4)

def compute_comet(scorer, references, predictions, sources):
    gpus = 1 if torch.cuda.is_available() else 0
    data = {"src": sources, "mt": predictions, "ref": references}
    data = [dict(zip(data, t)) for t in zip(*data.values())]
    score = scorer.predict(data, gpus=gpus, progress_bar=True)
    return score.system_score * 100

# Generate predictions for a given model and tokenizer
def predict(model, tokenizer, src_sequences, is_split_into_words, device, tgt_token, args):
    pred_sequences = []

    for i in tqdm(range(0, len(src_sequences), args.batch_size)):
        tokenized_batch = tokenizer(src_sequences[i:i + args.batch_size], return_tensors="pt", padding=True, truncation=True,
                                    is_split_into_words=is_split_into_words).to(device)

        if tgt_token is None:
            generated_ids = model.generate(**tokenized_batch, max_new_tokens=args.max_length, num_beams=5)
        else:
            generated_ids = model.generate(**tokenized_batch, max_new_tokens=args.max_length, num_beams=5,
                                           forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_token)
                                           )
        pred_sequences.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

    return pred_sequences

# Compute evaluation metrics for predictions
def compute_metrics(pred_sequences, tgt_sequences, src_sequences):
    metrics = {}
    pred_sequences, tgt_sequences = postprocess_text(pred_sequences, tgt_sequences)
    metrics['BLEU'] = metric_bleu.compute(predictions=pred_sequences, references=tgt_sequences)['score']
    metrics['chrF'] = metric_chrf.compute(predictions=pred_sequences, references=tgt_sequences)['score']
    metrics['METEOR'] = metric_meteor.compute(predictions=pred_sequences, references=tgt_sequences)['meteor']
    metrics['COMET'] = compute_comet(scorer=metric_comet, predictions=pred_sequences, references=tgt_sequences, sources=src_sequences)
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, nargs="+", required=False, help='Models for prediction')
    parser.add_argument('--errors', type=str, nargs="+", required=False, help='Error types to evaluate')
    parser.add_argument('--data-path', type=str, required=True, help='Evaluation data path')
    parser.add_argument('--save-path', type=str, required=False, help='Path for saving metrics')
    parser.add_argument('--splits', type=str, nargs='+', default=['test'], required=False, help='Data splits to evaluate')
    parser.add_argument('--src', type=str, required=True)
    parser.add_argument('--tgts', nargs="+", type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32, help='Inference batch size')
    parser.add_argument('--max-length', type=int, default=None, help='Maximum output sequence length')
    args = parser.parse_args()

    with open("./langcodes.json", "r") as f:
        langcodes = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.src == 'en':
        nlp = spacy.load("en_core_web_sm")
    elif args.src == 'fr':
        nlp = spacy.load("fr_core_news_sm")
    else:
        print(f"Language {args.src} is not supported.")
        return

    for tgt in args.tgts:
        direction = f"{args.src}-{tgt}"

        for model_name in args.models:
            model_prefix = model_name.split('/')[-1].split('-')[0].split('_')[0]
            model_langcodes = next((v for k, v in langcodes.items() if model_prefix in k), None)
            print(model_langcodes)
            
            src_token = model_langcodes[args.src] if model_langcodes else None
            tgt_token = model_langcodes[tgt] if model_langcodes else None

            config = AutoConfig.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config).to(device)

            if src_token:
                tokenizer.src_lang = src_token.strip("__")
            
            for split in args.splits:
                for type_ in args.errors:
                    
                    print(f"***** Evaluating {model_name} on {split} {type_} {direction} *****")

                    # Load evaluation data
                    test_df = pd.read_pickle(os.path.join(args.data_path, f"{split}.{type_}.{args.src}.pkl"))
                    clean_src_sequences = test_df.line.map(lambda x: [token.text for token in nlp(x.strip())]).tolist()
                    noisy_src_sequences = test_df.tokenized_line.tolist()

                    tgt_file = f"./data/europarl-st/{args.src}/{tgt}/{split}/segments.{tgt}"
                    with open(tgt_file, 'r') as f:
                        tgt_sequences = [line.strip() for line in f.readlines()]

                    # Set save path for metrics and predictions
                    if args.save_path is None:
                        save_path = os.path.join(model_name, f"metrics/{split}")
                    else:
                        save_path = os.path.join(args.save_path, f"{model_name.split('/')[-1]}/{direction}/{split}")
                        
                    Path(save_path).mkdir(parents=True, exist_ok=True)

                    clean_metrics_file = os.path.join(save_path, "translation_clean_metrics.json")
                    noisy_metrics_file = os.path.join(save_path, f"translation_{type_}_metrics.json")

                    # Compute metrics for clean data
                    clean_pred_sequences = predict(model, tokenizer, clean_src_sequences,
                                                   True, device, tgt_token, args)
                    clean_metrics = compute_metrics(clean_pred_sequences, tgt_sequences, clean_src_sequences)
                    save_metrics(clean_metrics_file, clean_metrics)
                    with open(os.path.join(save_path, 'clean_pred_sequences.txt'), 'w') as f:
                        f.writelines(seq + "\n" for seq in clean_pred_sequences)

                    # Compute metrics for noisy data
                    noisy_pred_sequences = predict(model, tokenizer, noisy_src_sequences,
                                                   True, device, tgt_token, args)
                    noisy_metrics = compute_metrics(noisy_pred_sequences, tgt_sequences, clean_src_sequences)
                    save_metrics(noisy_metrics_file, noisy_metrics)
                    with open(os.path.join(save_path, f"{type_}_pred_sequences.txt"), 'w') as f:
                        f.writelines(seq + "\n" for seq in noisy_pred_sequences)

                    # Calculate difference between clean and noisy metrics
                    diff_metrics = {m: clean_metrics[m] - noisy_metrics[m] for m in clean_metrics.keys()}
                    diff_metrics_file = os.path.join(save_path, f"{type_}_diff_metrics.json")
                    save_metrics(diff_metrics_file, diff_metrics)
                    print(f"***** Difference: COMET: {diff_metrics['COMET']} | BLEU: {diff_metrics['BLEU']} | chrF: {diff_metrics['chrF']} | METEOR: {diff_metrics['METEOR']} *****")
                    print(f"***** Clean COMET: {clean_metrics['COMET']} | Noisy COMET: {noisy_metrics['COMET']} *****")

if __name__ == "__main__":
    main()
