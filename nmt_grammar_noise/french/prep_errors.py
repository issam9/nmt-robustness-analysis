import numpy as np
import codecs
import spacy
import sys
from collections import defaultdict
import pandas as pd


nlp = spacy.load("fr_core_news_sm")

input_file = sys.argv[1]
with codecs.open(input_file, 'r', 'utf-8') as file:
    lines = file.readlines()

# Initialize counters and output storage
changes = 0 
output = defaultdict(list)

# List of French prepositions to look for
preps = ['à', 'après', 'avant', 'avec', 'chez', 'contre', 'dans', 'de', 'depuis', 'derrière', 'devant', 'durant', 'en', 'entre', 'envers', 'environ', 'jusque', 'malgré', 'par', 'parmi', 'pendant', 'pour', 'sans', 'sauf', 'selon', 'sous', 'suivant', 'sur', 'vers']

for idx, line in enumerate(lines):
    tokenized_line = [token.text for token in nlp(line.strip())]
    
    # Count occurrences of each preposition in the line
    prep_count = {prep: tokenized_line.count(prep) + tokenized_line.count(prep.capitalize()) for prep in preps}
    
    label = 'clean'
    label_ids = [0] * len(tokenized_line)
    word = None 
    index = None
    
    # Total number of prepositions in the line
    all_prep_count = sum(prep_count.values())
    
    if all_prep_count > 0:
        # Calculate probabilities for each preposition
        ps = [prep_count[k] / all_prep_count for k in preps]
        chosen_prep = np.random.choice(preps, 1, p=ps)[0]
        
        # Find positions of the chosen preposition in the tokenized line
        pos = [i for i, prep in enumerate(tokenized_line) if prep.lower() == chosen_prep.lower()]
        
        # Randomly select one of these positions
        ind = np.random.choice(pos, 1)[0]
        
        # Remove the chosen preposition from the list and select a substitute
        preps.remove(tokenized_line[ind].lower())
        subst = np.random.choice(preps, 1)[0]
        
        # Capitalize the substitute if the original preposition was capitalized
        if tokenized_line[ind][0] == tokenized_line[ind][0].upper():
            subst = subst.capitalize()
        
        # Replace the preposition in the tokenized line
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
