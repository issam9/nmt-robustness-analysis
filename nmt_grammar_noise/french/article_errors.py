import numpy as np
import codecs
import spacy
import sys
from collections import defaultdict
import pandas as pd

nlp = spacy.load("fr_core_news_sm") 

filename2 = sys.argv[1]
with codecs.open(filename2, 'r', 'utf-8') as inp:
    lines = inp.readlines()

changes = 0
output = defaultdict(list)  # Dictionary to store output data
for idx, line in enumerate(lines):
    tokenized_line = [token.text for token in nlp(line.strip())] # Tokenize 

    articles = ['la', 'le', 'une', 'un', 'les', 'des']
    # Count occurrences of each article in the tokenized line
    article_count = {article: tokenized_line.count(article)+tokenized_line.count(article.capitalize()) for article in articles}
    
    label = 'clean'
    label_ids = [0] * len(tokenized_line) 
    word = None
    index = None

    all_article_count = sum(article_count.values())  # Total occurrences of all articles
    if all_article_count > 0:
        # Calculate the probability distribution for article selection
        ps = [article_count[k] / all_article_count for k in articles]
        # Choose an article based on the distribution
        chosen_article = np.random.choice(articles, 1, p=ps)[0]

        # Get the positions of the chosen article in the tokenized line
        pos = [i for i, article in enumerate(tokenized_line) if article.lower() == chosen_article.lower()]
        
        # Randomly select one occurrence of the article to substitute
        ind = np.random.choice(pos, 1)[0]
        
        # Remove the chosen article from the list to ensure substitution
        articles.remove(tokenized_line[ind].lower())
        # Sample the new substitution article
        subst = np.random.choice(articles, 1)[0]

        # Capitalize the substitution if the original was capitalized
        if tokenized_line[ind][0] == tokenized_line[ind][0].upper():
            subst = subst.capitalize()
        tokenized_line[ind] = subst  # Apply the substitution
        label = 'noisy'  # Mark the label as a substitution
        label_ids[ind] = 1
        word = subst
        index = ind
        
        changes += 1  # Increment the change counter

    assert len(tokenized_line) == len(label_ids)  # Ensure the tokenized line and label IDs match
    
    # Store the results for the current line
    output['line'].append(line)
    output['tokenized_line'].append(tokenized_line)
    output['word'].append(word)
    output['index'].append(index)
    output['label'].append(label)
    output['label_ids'].append(label_ids)

# Convert the output to a DataFrame and save it as a pickle file
df = pd.DataFrame.from_dict(output)
df.to_pickle(sys.argv[2])

print("Changed:", changes, "out of", len(lines), "lines.")
