import numpy as np
import codecs
import spacy
import sys
from collections import defaultdict
import pandas as pd
from pattern.fr import singularize, pluralize


nlp = spacy.load("fr_core_news_sm")

input_file = sys.argv[1]
with codecs.open(input_file, 'r', 'utf-8') as file:
    lines = file.readlines()

# Read the file containing the noun positions
positions_file = sys.argv[2]
with codecs.open(positions_file, 'r', 'utf-8') as file:
    positions = [line.strip() for line in file.readlines()]

# Parse the noun positions into a list of dictionaries
noun_pos = []
for line in positions:
    pos = defaultdict(list)
    for i, item in enumerate(line.split(",")):
        item = item.strip()
        if item:
            pos[item].append(i)
    noun_pos.append(pos)
    

changes = 0 
output = defaultdict(list)

for idx, line in enumerate(lines):
    tokenized_line = [token.text for token in nlp(line.strip())]
    noun_count = defaultdict(int)
    noun_numbers = ["Plur", "Sing"]
    
    # Count the number of plural and singular nouns in the line
    for noun_number in noun_numbers:
        noun_count[noun_number] += 1 if noun_pos[idx].get(noun_number, None) is not None else 0
        
    label = 'clean'
    label_ids = [0] * len(tokenized_line)
    word = None 
    index = None
    
    # Determine the chosen noun number based on the counts
    all_noun_count = sum(noun_count.get(noun_number, 0) for noun_number in noun_numbers)
    if all_noun_count > 0:
        ps = [noun_count.get(noun_number, 0) / all_noun_count for noun_number in noun_numbers]
        chosen_noun_number = np.random.choice(noun_numbers, 1, p=ps)[0]
        
        pos = noun_pos[idx].get(chosen_noun_number, None)
        
        # Choose a random position for modification
        ind = np.random.choice(pos, 1)[0]
        
        if len(tokenized_line[ind]) > 1:
            backup = tokenized_line[ind]
            
            # Modify the token based on the chosen noun number
            if chosen_noun_number == "Plur":
                tokenized_line[ind] = singularize(tokenized_line[ind])
                label = 'false_sg'
                label_ids[ind] = noun_numbers.index("Plur") + 1
            else:
                tokenized_line[ind] = pluralize(tokenized_line[ind])
                label = 'false_pl'
                label_ids[ind] = noun_numbers.index("Sing") + 1
                
            word = tokenized_line[ind]
            index = ind
            
            # Revert the change if inflection fails
            if tokenized_line[ind] is None or tokenized_line[ind] == backup:
                label = 'clean'
                tokenized_line[ind] = backup
                label_ids[ind] = 0
                word = None 
                index = None
                changes -= 1
                
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
df.to_pickle(sys.argv[3])

print(f"Changed: {changes} out of {len(lines)} lines.")
