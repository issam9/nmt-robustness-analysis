scriptdir=./nmt-grammar-noise/french
data=./data/europarl-st/fr

src="fr"
for l in "es"
do
	outdir=./data/grammar-noise/$src-$l
	mkdir -p $outdir

	for f in dev test train
	do
        echo "---------- Language pair: $src-$l | Set: $f ------------"

		inp=$data/$l/$f/segments.$src
        outdir=./data/grammar-noise/$src-$l
		mkdir -p $outdir/positions-$l


        outp=$outdir/$f.nounnum.$src.pkl
        python $scriptdir/identify_nounnum.py $inp $outdir/positions-$l/$f.nounnum
        python $scriptdir/nounnum_errors.py $inp $outdir/positions-$l/$f.nounnum $outp

        outp=$outdir/$f.article.$src.pkl
        python $scriptdir/article_errors.py $inp $outp

        outp=$outdir/$f.prep.$src.pkl
        python $scriptdir/prep_errors.py $inp $outp

	done

done




