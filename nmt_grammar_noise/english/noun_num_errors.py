import numpy as np
import codecs
import inflection
import sys
import spacy
import pandas as pd
from collections import defaultdict

nlp = spacy.load("en_core_web_sm")

# We sample uniformly
false_plural_prob = 0.5
false_singular_prob = 0.5

# Read input files
filename2 = sys.argv[1]
with codecs.open(filename2, 'r', 'utf-8') as file:
    lines = file.readlines()

with codecs.open(sys.argv[2], 'r', 'utf-8') as file:
    sng_nouns = [[int(x) for x in line.strip().split()] for line in file.readlines()]

with codecs.open(sys.argv[3], 'r', 'utf-8') as file:
    pl_nouns = [[int(x) for x in line.strip().split()] for line in file.readlines()]

# Initialize output and change counter
output = defaultdict(list)
changes = 0

for idx, line in enumerate(lines):
    tokenized_line = [token.text for token in nlp(line.strip())]
    
    # Count the number of singular and plural nouns
    npl = len(pl_nouns[idx])
    nsng = len(sng_nouns[idx])

    label = 'clean'
    label_ids = [0] * len(tokenized_line)
    word = None
    index = None
    
    # If there are any nouns, potentially introduce errors
    if npl + nsng:
        # Calculate the probability distribution for errors
        tempsum = float(npl * false_singular_prob + nsng * false_plural_prob)
        is_pl_vals = [1, 0]
        ps = [npl * false_singular_prob / tempsum, nsng * false_plural_prob / tempsum]
        
        # Sample whether to introduce a plural or singular error
        pl = np.random.choice(is_pl_vals, 1, p=ps)[0]

        # Sample a position for the error
        pos = pl_nouns[idx] if pl else sng_nouns[idx]
        ind = np.random.choice(pos, 1)[0]
        
        if tokenized_line[ind]:
            if pl:
                tokenized_line[ind] = inflection.singularize(tokenized_line[ind])
                label = 'false_sg'
                label_ids[ind] = 1
            else:
                tokenized_line[ind] = inflection.pluralize(tokenized_line[ind])
                label = 'false_pl'
                label_ids[ind] = 2
            word = tokenized_line[ind]
            index = ind
            changes += 1

    # Ensure tokenized line length matches label IDs length
    assert len(tokenized_line) == len(label_ids)
    
    output['line'].append(line)
    output['tokenized_line'].append(tokenized_line)
    output['word'].append(word)
    output['index'].append(index)
    output['label'].append(label)
    output['label_ids'].append(label_ids)

df = pd.DataFrame.from_dict(output)
df.to_pickle(sys.argv[4])

print(f"Changed: {changes} out of {len(lines)} lines.")
