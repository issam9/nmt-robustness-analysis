import benepar, spacy
import sys

spacy.prefer_gpu(1)

benepar.download('benepar_en3')
nlp = spacy.load('en_core_web_md')
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

# Read input lines from the file
with open(sys.argv[1], 'r') as f:
    lines = [line.rstrip() for line in f.readlines()]

# Process each line and extract its first sentence's parse tree
docs = [nlp(line) for line in lines]
sents = [list(doc.sents)[0] for doc in docs]
parses = [sent._.parse_string for sent in sents]

# Write parse trees to the output file
with open(sys.argv[2], 'w') as f:
    for line in parses:
        f.write(line + "\n")
