
scriptdir=./nmt-grammar-noise/english
data=./data/europarl-st/en


for l in "es" "de" "it" "nl"
do
	outdir=./data/grammar-noise/en-$l
	mkdir -p $outdir
	# Iterate over the filenames 
	for f in test dev train
	do
        mkdir ./data/grammar-noise/en-$l/parses
		inp=$data/$l/$f/segments.en
		parsed=./data/grammar-noise/en-$l/parses/$f.segments.en.parsed

		python $scriptdir/parse.py $inp $parsed

	done
done




