import sys
from nltk.tree import Tree
import codecs

# Function to find indices of singular noun words (NN) in the tree
def find_nn_indices(tree):
    nn_indices = []
    for index, subtree in enumerate(tree):
        # Check if the subtree label is 'NN' (singular noun)
        if subtree[1] == 'NN':
            nn_indices.append(index)
    return nn_indices

def main():
    with open(sys.argv[2], 'w') as outp:
        with codecs.open(sys.argv[1], 'r', 'utf-8') as f:
            inp = [line.strip() for line in f.readlines()]
        
        for line in inp:
            # Convert the line to a syntactic tree
            tree = Tree.fromstring(line)
            # Find indices of singular nouns in the tree
            nn_indices = find_nn_indices(tree.pos())
            # Write indices to the output file, separated by spaces
            if nn_indices:
                outp.write(" ".join(map(str, nn_indices)) + '\n')
            else:
                outp.write('\n')

if __name__ == "__main__":
    main()
