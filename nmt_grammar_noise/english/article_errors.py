import numpy as np
from collections import defaultdict
import codecs
import sys
import spacy
import pandas as pd


nlp = spacy.load("en_core_web_sm")

with codecs.open("./nmt-grammar-noise/official-2014.combined-withalt.m2", 'r', 'utf-8') as file:
    lines = file.readlines()

# Initialize a dictionary to store counts of substitutions
dets = defaultdict(lambda: 0.0)

# Parse the file and collect counts for each substitution type
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
        
        if errortype == "ArtOrDet":
            if end == start:
                inw = ''
            elif end == start + 1:
                inw = last_sent[start].lower()
            else:
                inw = ' '.join(last_sent[start:end]).lower()
            dets[subst, inw] += 1

# Create a confusion matrix-like structure to store total counts
detlist = ['a', 'an', 'the']
detsum = defaultdict(lambda: 0.0)

# Sum counts of substitutions for each article type
for d in detlist:
    for e in detlist:
        if e != d and e != '':
            detsum[d] += dets[d, e]

detvals = defaultdict(lambda: [])
detps = defaultdict(lambda: [])

# Store possible substitutions and their probabilities
for d in detlist:
    for e in detlist:
        if dets[d, e] and d != e and e != '':
            detvals[d].append(e)
            detps[d].append(dets[d, e] / detsum[d])

# Initialize probabilities for substitution sampling
detpps = {}
for d in detlist:
    detpps[d] = 1.0 / len(detlist)

# Read the input file containing text to process
input_file = sys.argv[1]
with codecs.open(input_file, 'r', 'utf-8') as file:
    lines = file.readlines()

print("Read all files successfully")

# Function to capitalize the first letter of a word
def capitalize(s):
    return s[0].upper() + s[1:] if s else s

# Process each line and introduce errors based on probabilities
output = defaultdict(list)
changes = 0  # Counter lines changed

for idx, line in enumerate(lines):
    tokenized_line = [token.text for token in nlp(line.strip())]
    
    # Count occurrences of each article in the line
    ays = tokenized_line.count('a') + tokenized_line.count('A')
    ans = tokenized_line.count('an') + tokenized_line.count('An')
    thes = tokenized_line.count('the') + tokenized_line.count('The')

    label = 'clean'
    label_ids = [0] * len(tokenized_line)
    word = None
    index = None
    
    if ays + ans + thes:
        # Calculate probabilities for each article
        tempsum = float(ays * detpps['a'] + ans * detpps['an'] + thes * detpps['the'])
        vals = ['a', 'an', 'the']
        ps = [ays * detpps['a'] / tempsum, ans * detpps['an'] / tempsum, thes * detpps['the'] / tempsum]
        
        # Sample an article to substitute
        ch = np.random.choice(vals, 1, p=ps)[0]
        subst = np.random.choice(detvals[ch], 1, p=detps[ch])[0]
        
        # Find all positions of the chosen article
        pos = [index for index, word in enumerate(tokenized_line) if word.lower() == ch]
        ind = np.random.choice(pos, 1)[0]
        
        # Capitalize the substitute if needed
        if tokenized_line[ind][0].isupper():
            subst = capitalize(subst)
        
        # Replace the article and update labels
        tokenized_line[ind] = subst
        label = 'noisy'
        label_ids[ind] = 1
        word = subst
        index = ind
        changes += 1

    # Ensure the length of tokenized_line matches label_ids
    assert len(tokenized_line) == len(label_ids)
    
    output['line'].append(line)
    output['tokenized_line'].append(tokenized_line)
    output['word'].append(word)
    output['index'].append(index)
    output['label'].append(label)
    output['label_ids'].append(label_ids)

# Save the output to a pickle file
df = pd.DataFrame.from_dict(output)
df.to_pickle(sys.argv[2])

print(f"Changed: {changes} out of {len(lines)} lines.")
