import spacy
import sys
from spacy_lefff import LefffLemmatizer, POSTagger
from spacy.language import Language

nlp = spacy.load('fr_core_news_sm') 

def create_french_lemmatizer(nlp, name):
    return LefffLemmatizer(after_melt=True)

def create_french_pos(nlp, name):
    return POSTagger()

# Register the custom lemmatizer and POS tagger as spaCy components
Language.factory("french_lemmatizer", func=create_french_lemmatizer)
Language.factory("french_pos", func=create_french_pos)

# Add the POS tagger and lemmatizer to the pipeline
nlp.add_pipe("french_pos", name='pos_lefff')
nlp.add_pipe("french_lemmatizer", name='lefff')

with open(sys.argv[1], 'r') as f:
    lines = [line.rstrip() for line in f.readlines()]
print(len(lines))

all_nouns = []
for line in lines:
    doc = nlp(line)
    nouns = []
    # For each token, check if it's a noun ("NC" in melt_tagger) and if it has a "Number" attribute
    for token in doc:
        if token._.melt_tagger == "NC" and len(token.morph.get("Number")) > 0:                        
            nounnum = token.morph.get("Number")[0]
            nouns.append(nounnum)  # Append noun number (singular/plural)
        else:
            nouns.append("")  # If not a noun, append an empty string
    all_nouns.append(nouns)

# Write the list of noun numbers to the output file, joining with commas
with open(sys.argv[2], 'w') as f:
    for line in all_nouns:
        f.write(",".join(line) + "\n")
