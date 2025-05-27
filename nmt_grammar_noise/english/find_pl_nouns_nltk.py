import sys
from nltk.tree import Tree
import codecs

# Function to find indices of plural noun words (NNS) in the tree
def find_nns_indices(tree):
    nns_indices = []
    for index, subtree in enumerate(tree):
        # Check if the subtree label is 'NNS' (plural noun)
        if subtree[1] == 'NNS':
            nns_indices.append(index)
    return nns_indices

def main():
    with open(sys.argv[2], 'w') as outp:
        with codecs.open(sys.argv[1], 'r', 'utf-8') as f:
            inp = [line.strip() for line in f.readlines()]
        
        for line in inp:
            # Convert the line to a syntactic tree
            tree = Tree.fromstring(line)
            # Find indices of plural nouns in the tree
            nns_indices = find_nns_indices(tree.pos())
            # Write indices to the output file, separated by spaces
            if nns_indices:
                outp.write(" ".join(map(str, nns_indices)) + '\n')
            else:
                outp.write('\n')

if __name__ == "__main__":
    main()
