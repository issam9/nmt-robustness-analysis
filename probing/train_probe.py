import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm, trange

from probe_dataset import ProbeDataset
from probes import LinearClassifier
from utils import save_model, load_model, seed, compute_metrics, checkpoint



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(train_loader, dev_loader, model, epochs, patience, label2id, output_path, args):

    model = model.to(DEVICE)

    id2label = {idx: label for label, idx in label2id.items()}

    # training loop
    optimizer = optim.Adam(model.parameters(), weight_decay=args.wd)

    best_metric = -1
    eps = 0.002

    epoch_iter = trange(0, epochs)
    for epoch_num in epoch_iter:
        model = model.train()
        train_loss = 0

        for batch in tqdm(train_loader):
            x, y = batch
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            
            loss, logits, labels = model(x, y)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        train_loss = train_loss / len(train_loader)
        print("Train loss: {}".format(train_loss))

        # evaluation loop
        model = model.eval()
        eval_loss = 0
        
        eval_predictions, eval_labels = [], []
        for batch in tqdm(dev_loader):
            x, y = batch
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            with torch.no_grad():
                loss, logits, labels = model(x, y)

                preds = torch.argmax(F.log_softmax(logits, dim=2), dim=2)

                eval_predictions.extend(preds)
                eval_labels.extend(labels)

            eval_loss += loss.item()

        eval_loss = eval_loss / len(dev_loader)

        # move to cpu
        eval_predictions = [y_pred.detach().cpu().numpy() for y_pred in eval_predictions]
        eval_labels = [y_true.detach().cpu().numpy() for y_true in eval_labels]

        # flatten
        y_pred = [p_i for p in eval_predictions for p_i in p]
        y_true = [l_i for l in eval_labels for l_i in l]
        
        assert len(y_pred) == len(y_true)

        # reduce to ignore padded items
        y_pred = [id2label[p] for p, t in zip(y_pred, y_true) if t != -1]
        y_true = [id2label[t] for t in y_true if t != -1]

        assert len(y_pred) == len(y_true)

        metrics = compute_metrics(y_pred, y_true, list(label2id.keys()), verbose=False)
        metrics[f'epoch_{epoch_num}_train_loss'] = train_loss
        metrics[f'epoch_{epoch_num}_eval_loss'] = eval_loss

        epoch_iter.set_description(
            'Train loss: %.4f Dev loss: %.4f F1: %.4f Binary F1: %.4f' % (train_loss, eval_loss, metrics['f1-errors'], metrics['binarized-f1']))

        # set best
        eval_metric = metrics['f1-errors']
        if eval_metric > best_metric + eps:
            best_metric = eval_metric
            best_epoch = epoch_num

            # save best model
            save_model(output_path, model, metrics, 'best_epoch', id2label, args)

        # Check-pointing to save the current epoch
        checkpoint(output_path, model, metrics, f'epoch_{epoch_num}', id2label, args)

        # check patience     
        if epoch_num - best_epoch >= patience:
            # end training early
            print(f'Patience reached. Ending training at epoch {epoch_num}')
            break

    model, _ = load_model(os.path.join(output_path, 'best_epoch'))
    
    return model


def main(args):
    seed(args.seed)
    
    # Load train and dev encodings and labels
    train_encodings = np.load(os.path.join(args.data_path, f'{args.model_name}/{args.pooling}/train_probing.{args.error}.{args.src}.encoding.npy'), allow_pickle=True)
    train_labels = np.load(os.path.join(args.data_path, f'{args.model_name}/{args.pooling}/train_probing.{args.error}.{args.src}.label_ids.npy'), allow_pickle=True)
    dev_encodings = np.load(os.path.join(args.data_path, f'{args.model_name}/{args.pooling}/dev.{args.error}.{args.src}.encoding.npy'), allow_pickle=True)
    dev_labels = np.load(os.path.join(args.data_path, f'{args.model_name}/{args.pooling}/dev.{args.error}.{args.src}.label_ids.npy'), allow_pickle=True)
    label2id = np.load(os.path.join(args.data_path, f'{args.model_name}/{args.pooling}/train_probing.{args.error}.{args.src}.label2id.npy'), allow_pickle=True).item()

    # Transpose encodings for easier layer-wise access (assumed shape [layers, samples, features])
    train_encodings = train_encodings.transpose(1, 0)
    dev_encodings = dev_encodings.transpose(1, 0)

    # Loop over each layer in the encodings
    for i in range(train_encodings.shape[0]):
        # Prepare dataset and dataloaders for the current layer
        train_dataset = ProbeDataset(encodings=train_encodings[i], labels=train_labels, label2id=label2id)
        dev_dataset = ProbeDataset(encodings=dev_encodings[i], labels=dev_labels, label2id=label2id)
        
        # Create DataLoader for train and dev datasets with padding
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            collate_fn=lambda batch: (
                pad_sequence([item[0] for item in batch], batch_first=True, padding_value=-1),
                pad_sequence([item[1] for item in batch], batch_first=True, padding_value=-1)
            )
        )
        
        dev_loader = DataLoader(
            dev_dataset, 
            batch_size=args.batch_size, 
            collate_fn=lambda batch: (
                pad_sequence([item[0] for item in batch], batch_first=True, padding_value=-1),
                pad_sequence([item[1] for item in batch], batch_first=True, padding_value=-1)
            )
        )

        # Get the hidden size (number of features) for this layer
        hidden_size = train_encodings[i][0].shape[-1]

        # Initialize a linear classifier for this layer's hidden representations
        model = LinearClassifier(hidden_size=hidden_size, num_labels=len(label2id), dropout=args.dropout)

        # Define path to save the trained model
        save_path = os.path.join(args.save_path, f"{args.model_name}/layer-{i}/{args.pooling}/{args.error}")
        
        # Train the model on the current layer and save it
        model = train(train_loader, dev_loader, model, args.epochs, args.patience, label2id, save_path, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data-path", type=str, required=True, help='Path to folder containing train and dev encodings')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--model-name', type=str, required=True, help='Name of the model used for encoding')
    parser.add_argument("--error", type=str, required=True, help='Type of error to probe')
    parser.add_argument("--pooling", type=str, default='mean', help='Pooling method used for encoding (e.g., mean, max)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate for the classifier')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--wd', type=float, default=0.0001, help='Weight decay for optimization')
    parser.add_argument("--save-path", type=str, required=True, help='Directory to save trained models')
    parser.add_argument("--seed", type=int, default=123, help='Random seed for reproducibility')

    args = parser.parse_args()
    main(args)
