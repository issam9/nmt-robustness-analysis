import json 
import os
import numpy as np
import torch
import random
import logging
from sklearn.metrics import f1_score, precision_recall_fscore_support
import glob
from probes import LinearClassifier


def seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    

def compute_metrics(y_pred, y_true, label_names, verbose=False):
    error_labels = label_names.copy()
    error_labels.remove('clean')
    # Only care whether we detected the error or not
    binarized_y_true = ['clean' if y=='clean' else 'noisy' for y in y_true]
    binarized_y_pred = ['clean' if y=='clean' else 'noisy' for y in y_pred]
    
    precision_all, recall_all, f1_all , _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    precision_errors, recall_errors, f1_errors , _ = precision_recall_fscore_support(y_true, y_pred, average='macro', labels=error_labels, zero_division=0.)
    
    b_precision, b_recall, b_f1 , _ = precision_recall_fscore_support(binarized_y_true, binarized_y_pred, average='binary', pos_label='noisy')
    
    metrics =  {"f1-all": f1_all,
                "f1-errors": f1_errors, 
                'precision-all': precision_all, 
                'recall-all': recall_all, 
                'precision-errors': precision_errors, 
                'recall-errors': recall_errors,
                'binarized-f1': b_f1,
                'binarized-precision': b_precision,
                'binarized-recall': b_recall
                }
    
    return metrics
        
    
def save_metrics(target_file, metrics):
    with open(target_file, 'w') as f:
        json.dump(metrics, f, indent=4)

def save_model(output_path, model, metrics, model_label, id2label, args):
    target_folder = os.path.join(output_path, model_label)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # save label dictionary
    np.save(os.path.join(target_folder, 'idx2labels.npy'), id2label)

    # save metrics
    save_metrics(os.path.join(target_folder, 'metrics.json'), metrics)

    with open(os.path.join(target_folder, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    # save model
    fname = os.path.join(target_folder, 'model.pt')
    torch.save({
        'kwargs': model.get_args(),
        'model_state_dict': model.state_dict(),
    }, fname)
    

def checkpoint(output_path, model, metrics, model_label, id2label, args):

    # save new epoch
    save_model(output_path, model, metrics, model_label, id2label, args)

    # save a copy of metrics to top level folder
    with open(os.path.join(output_path, f'{model_label}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    # remove old epoch
    first, epoch_num = model_label.split('_')
    epoch_num = int(epoch_num)
    if epoch_num > 0:
        old_epoch_path = os.path.join(output_path, f'{first}_{epoch_num-1}')
        if os.path.exists(old_epoch_path):
            last_epoch_files = glob.glob(os.path.join(old_epoch_path, '*'))
            for f in last_epoch_files:
                os.remove(f)
            os.removedirs(old_epoch_path)
            

def load_model(saved_path):
    fname = os.path.join(saved_path, 'model.pt')
    checkpoint = torch.load(fname)
    
    model = LinearClassifier(**checkpoint['kwargs'])
    model.load_state_dict(checkpoint['model_state_dict'])

    # load label dictionary
    id2label = np.load(os.path.join(saved_path, 'idx2labels.npy'), allow_pickle=True).item()
    
    return model, id2label