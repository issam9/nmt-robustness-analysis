
scriptdir=./nmt-grammar-noise/english
data=./data/europarl-st/en

for l in "es" "de" "it" "nl"
do
	outdir=./data/grammar-noise/en-$l
	mkdir -p $outdir
	mkdir -p $outdir/parses 

	# Iterate over the filenames 
	for f in dev test train
	do
        echo "---------- Language pair: en-$l | Set: $f ------------"

		inp=$data/$l/$f/segments.en
        outdir=./data/grammar-noise/en-$l
        parsed=./data/grammar-noise/en-$l/parses/$f.segments.en.parsed
		mkdir -p $outdir/positions-$l

        # Nouns
        outp=$outdir/$f.nounnum.en.pkl
        python $scriptdir/find_sng_nouns_nltk.py $parsed $outdir/positions-$l/$f.sngins 
        python $scriptdir/find_pl_nouns_nltk.py $parsed $outdir/positions-$l/$f.plins 
        python $scriptdir/noun_num_errors.py $inp $outdir/positions-$l/$f.sngins $outdir/positions-$l/$f.plins $outp

        # Articles
        outp=$outdir/$f.article.en.pkl
        python $scriptdir/article_errors.py $inp $outp

        # Prepositions
        outp=$outdir/$f.prep.en.pkl
        python $scriptdir/prep_errors.py $inp $outp

	done

done