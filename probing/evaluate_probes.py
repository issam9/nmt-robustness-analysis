from utils import load_model, compute_metrics, save_metrics
import numpy as np
import os
from probe_dataset import ProbeDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import torch
import torch.nn.functional as F
import argparse
import re


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def predict(dataloader, model, id2label):

    model = model.to(DEVICE)
    
    model = model.eval()
    loss = 0
    
    preds = []
    labels = []
    probs = []
    for batch in tqdm(dataloader):
        x, y = batch
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        with torch.no_grad():
            batch_loss, logits, batch_labels = model(x, y)

            batch_probs = F.softmax(logits, dim=2).float()
            batch_preds = torch.argmax(F.log_softmax(logits, dim=2), dim=2)

            probs.extend(batch_probs)
            preds.extend(batch_preds)
            labels.extend(batch_labels)

        loss += batch_loss.item()

    loss = loss / len(dataloader)

    probs = [y_prob.detach().cpu().numpy() for y_prob in probs]
    preds = [y_pred.detach().cpu().numpy() for y_pred in preds]
    labels = [y_true.detach().cpu().numpy() for y_true in labels]

    y_probs = [p_i for p in probs for p_i in p]
    y_pred = [p_i for p in preds for p_i in p]
    y_true = [l_i for l in labels for l_i in l]
    
    assert len(y_pred) == len(y_true) == len(y_probs)

    y_probs = [p for p, t in zip(y_probs, y_true) if t != -1]
    y_pred = [id2label[p] for p, t in zip(y_pred, y_true) if t != -1]
    y_true = [id2label[t] for t in y_true if t != -1]

    assert len(y_pred) == len(y_true) == len(y_probs)

    return y_pred, y_true, y_probs, loss


def evaluate(model_path, encodings, labels, batch_size):
    model, id2label = load_model(model_path)
    label2id = {v:k for k, v in id2label.items()}
    
    test_dataset = ProbeDataset(encodings=encodings, labels=labels, label2id=label2id)
    test_loader = DataLoader(test_dataset, 
                                   batch_size=batch_size, 
                                   collate_fn=lambda batch: (
                                        pad_sequence([item[0] for item in batch], batch_first=True, padding_value=-1),
                                        pad_sequence([item[1] for item in batch], batch_first=True, padding_value=-1)
                                    )
                                )
    preds, labels, probs, test_loss = predict(test_loader, model, id2label)
    
    metrics = compute_metrics(y_pred=preds, y_true=labels, label_names=list(id2label.values()))
    metrics['test_loss'] = test_loss
        
    return metrics

def main(args):
    
    for model_name in args.models:
        for error in args.errors:        
            if error == 'morpheus':
                error = re.split(r"[-_]", model_name)[0]
            else:
                error = error    
            test_encodings = np.load(os.path.join(args.data_path, f'{model_name}/{args.pooling}/test.{error}.{args.src}.encoding.npy'), allow_pickle=True)
            test_labels = np.load(os.path.join(args.data_path, f'{model_name}/{args.pooling}/test.{error}.{args.src}.label_ids.npy'), allow_pickle=True)

            test_encodings = test_encodings.transpose(1, 0)
            
            for i in range(test_encodings.shape[0]):    
                model_path = os.path.join(args.probe_path, f"{model_name}/layer-{i}/{args.pooling}/{error}/best_epoch")
                metrics = evaluate(model_path, test_encodings[i], test_labels, args.batch_size)
                save_metrics(os.path.join(model_path, "test_metrics.json"), metrics)    
            

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help='Path to folder containing train and dev representations')
    parser.add_argument('--batch-size', type=int, required=False, default=128, help='Batch size')
    parser.add_argument('--models', nargs='+', required=False, type=str, help="Models we used for encodings")
    parser.add_argument("--errors", type=str, nargs='+', required=False, help='Type of the error we are probing')
    parser.add_argument("--probe-path", type=str, required=False, help='Path to probe model')
    parser.add_argument("--pooling", type=str, required=False, default='last', help='Pooling used for encoding')
    parser.add_argument('--src', default='en', help='Source language')
    args = parser.parse_args()
    
    main(args)