# Set to fr for French evaluation
src=en
for tgt in "es" "de" "nl" "it"
do
    for model in "opus" "mbart" "nllb" "m2m100"
    do
        for error in  "article" "nounnum" "prep" "morpheus"
        do
            python ./finetuning/evaluate_robustness.py --models outputs/models/$model-$src-$tgt-$error \
            outputs/models/$model-$src-$tgt-clean-$error --data-path ./data/grammar-noise/$src-$tgt --splits test --src $src --tgts $tgt --errors $error --batch-size 32
        done
    done
done


for tgt in "es" "de" "it" "nl"
do
    for error in "nounnum" "prep" "article" "morpheus"
    do
        python ./finetuning/evaluate_robustness.py --models facebook/nllb-200-distilled-600M facebook/mbart-large-50-many-to-many-mmt facebook/m2m100_418M Helsinki-NLP/opus-mt-$src-$tgt \
        --data-path ./data/grammar-noise/$src-$tgt --splits test --src $src --tgts $tgt --errors $error --save-path ./outputs/models/ \
        --batch-size 32
    done
done

