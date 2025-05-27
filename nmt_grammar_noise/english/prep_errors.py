import numpy as np
from collections import defaultdict
import codecs
import sys
import spacy
import pandas as pd

nlp = spacy.load("en_core_web_sm")

with codecs.open("./nmt-grammar-noise/official-2014.combined-withalt.m2", 'r', 'utf-8') as inp:
    lines = inp.readlines()

preps = defaultdict(lambda: 0.0)

# Parse file and collect counts of preposition errors
for line in lines:
    line = line.strip().split()
    if line and line[0] == 'S':
        last_sent = line[1:]
    elif line and line[0] == 'A':
        start = int(line[1])
        line2 = line[2].split('|||')
        end = int(line2[0])
        errortype = line2[1]
        subst = line2[2].lower() if line2[2] else ''
        if errortype == "Prep":
            if end == start:
                inw = ''
            elif end == start + 1:
                inw = last_sent[start].lower()
            else:
                inw = (' '.join(last_sent[start:end])).lower()
            preps[subst, inw] += 1

# Create confusion matrix
preplist = ['on', 'in', 'at', 'from', 'for', 'under', 'over', 'with', 'into', 'during', 'until', 'against', 'among', 'throughout', 'of', 'to', 'by', 'about', 'like', 'before', 'after', 'since', 'across', 'behind', 'but', 'out', 'up', 'down', 'off']
prepsum = defaultdict(lambda: 0.0)
for a in preplist:
    for b in preplist:
        if preps[a, b] and a != b and b != '':
            prepsum[a] += preps[a, b]

prepvals = defaultdict(lambda: [])
prepps = defaultdict(lambda: [])
for d in preplist:
    for e in preplist:
        if preps[d, e] and d != e and e != '':
            prepvals[d].append(e)
            prepps[d].append(preps[d, e] / prepsum[d])

# Uniform sampling from prepositions based on discovered errors
preppps = defaultdict(lambda: 0.0)
for d in prepps:
    preppps[d] = 1./len(preplist)

# Read input file 
filename2 = sys.argv[1]
with codecs.open(filename2, 'r', 'utf-8') as inp:
    lines = inp.readlines()

# Function to capitalize the first letter of a word
def capitalize(s):
    if s:
        s = s[0].upper() + s[1:]
    return s

# Write output file
changes = 0
output = defaultdict(list)
for idx, line in enumerate(lines):
    tokenized_line = [token.text for token in nlp(line.strip())]
    count_occs = [tokenized_line.count(d) + tokenized_line.count(capitalize(d)) for d in preplist]

    # Calculate the probability distribution of prepositions
    tempsum = np.sum([count_occs[i] * preppps[d] for i, d in enumerate(preplist)])
    label = 'clean'
    label_ids = [0] * len(tokenized_line)
    word, index = None, None

    # If prepositions are present
    if tempsum > 0:
        vals = preplist
        ps = [count_occs[i] * preppps[d] / tempsum for i, d in enumerate(preplist)]
        ch = np.random.choice(vals, 1, p=ps)[0]

        # Sample error substitution
        subst = np.random.choice(prepvals[ch], 1, p=prepps[ch])[0]
        pos = [index for index, word in enumerate(tokenized_line) if word == ch or word == capitalize(ch)]
        ind = np.random.choice(pos, 1)[0]

        # Apply substitution
        if tokenized_line[ind][0] == tokenized_line[ind][0].upper():
            subst = capitalize(subst)
        tokenized_line[ind] = subst

        label = 'noisy'
        label_ids[ind] = 1
        word = subst
        index = ind
        changes += 1

    assert len(tokenized_line) == len(label_ids)

    output['line'].append(line)
    output['tokenized_line'].append(tokenized_line)
    output['word'].append(word)
    output['index'].append(index)
    output['label'].append(label)
    output['label_ids'].append(label_ids)

df = pd.DataFrame.from_dict(output)
df.to_pickle(sys.argv[2])

print("Changed: ", changes, " out of ", len(lines), " lines.")
