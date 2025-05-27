#!/bin/bash

# Introduce errors in the English data
./nmt_grammar_noise/english/parse.sh        # Parse to find noun number
./nmt_grammar_noise/english/introduce_errors.sh

# Intorduce errors in the French data
./nmt_grammar_noise/french/introduce_errors.sh

# Morpheus Attack
./nmt_grammar_noise/morpheus_attack.sh