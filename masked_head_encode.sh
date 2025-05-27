# Set to fr for French encodings
src=en

for tgt in "es" "de" "nl" "it"
do
    for error in "prep" "nounnum" "article" "morpheus"
    do
        for model in "opus" "mbart" "nllb" "m2m100"
        do
            python representations/masked_head_encode.py --src-lang $src --tgt-lang $tgt --models outputs/models/$model-$src-$tgt-$error \
            outputs/models/$model-$src-$tgt-clean-$error --error $error --pooling last --batch-size 32 --max-length 1024
        done
    done
done

for tgt in "es" "de" "nl" "it"
do
    for error in "prep" "nounnum" "article" "morpheus"
    do
        python representations/masked_head_encode.py --src-lang $src --tgt-lang $tgt --models facebook/nllb-200-distilled-600M facebook/mbart-large-50-many-to-many-mmt \
        facebook/m2m100_418M Helsinki-NLP/opus-mt-$src-$tgt --error $error --pooling last --batch-size 32 --max-length 1024
    done
done
