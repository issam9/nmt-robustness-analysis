#!/bin/bash

src=en
data=./data/europarl-st/$src


for tgt in "es" "de" "it" "nl"
do
    outdir=./data/grammar-noise/$src-$tgt/morpheus
    mkdir -p $outdir
    models=(facebook/m2m100_418M Helsinki-NLP/opus-mt-$src-$tgt facebook/mbart-large-50-many-to-many-mmt facebook/nllb-200-distilled-600M)

    for model in ${models[@]}
    do
        prefix=$(echo "$model" | cut -d '/' -f 2 | cut -d '-' -f 1 | cut -d '_' -f 1)
        # Iterate over the filenames 
        for split in dev test
            do
                echo "---------- Language pair: $src-$tgt | Set: $split | Model: $prefix ------------"
                if [ $prefix == "opus" ]
                then
                    python ./nmt_grammar_noise/morpheus_attack.py --src-lang $src --tgt-lang $tgt --source-file $data/$tgt/$split/segments.$src \
                    --target-file $data/$tgt/$split/segments.$tgt --output-file $outdir/$split.$prefix.$src.pkl --model $model --batch-size 128 --multi
                else
                    echo "Using multilingual model"
                    python ./nmt_grammar_noise/morpheus_attack.py --src-lang $src --tgt-lang $tgt --source-file $data/$tgt/$split/segments.$src \
                    --target-file $data/$tgt/$split/segments.$tgt --output-file $outdir/$split.$prefix.$src.pkl --model $model --multilingual --batch-size 128 -multi
                fi
            done
        split=train
        python ./nmt_grammar_noise/random_inflect.py --src-lang $src --source-file $data/$tgt/$split/segments.$src --output-file $outdir/$split.$prefix.$src.pkl --counts-file $outdir/dev.$prefix.$src.pkl
    done
done