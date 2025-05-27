from train_probe import main as train_probe
import argparse
import re

def main(args):
    for model_name in args.models:
        for error in args.errors:
            args.model_name = model_name
            if error == 'morpheus':
                args.error = re.split(r"[-_]", model_name)[0]
            else:
                args.error = error
            train_probe(args)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help='Path to folder containing train and dev representations')
    parser.add_argument('--batch-size', type=int, required=False, default=32)
    parser.add_argument('--models', nargs='+', required=False, type=str, help="Models we used for encodings")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--errors", type=str, nargs='+', required=False, help='Type of the error we are probing')
    parser.add_argument("--pooling", type=str, required=False, default='last', help='The type of pooling that was used to encode the data')
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs")
    parser.add_argument('--patience', type=int, default=10, help="Patience for early stopping")
    parser.add_argument('--wd', type=float, default=0.0001, help="Weight decay")
    parser.add_argument("--save-path", type=str, required=True, help='Path to folder to save model')
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument('--src', default='en', help="Source language")
    args = parser.parse_args()
            
    main(args)