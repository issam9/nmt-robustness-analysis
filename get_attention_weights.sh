# Set to fr for French attention weights
src=en

for tgt in "es" "de" "it" "nl"
do
    for error in "article" "nounnum" "prep"
    do
        for model in  m2m100 mbart nllb opus
        do 
            python representations/get_attention_weights.py --data-dir ./data/grammar-noise/$src-$tgt --src-lang $src --tgt-lang $tgt --models outputs/finetuning/$model-$src-$tgt-$error \
            outputs/finetuning/$model-$src-$tgt-clean-$error --error $error --batch-size 32 --max-length 1024
        done
    done
done


for tgt in "es" "de" "it" "nl"
do
    for error in "article" "nounnum" "prep"
    do
        python representations/get_attention_weights.py --data-dir ./data/grammar-noise/$src-$tgt --src-lang $src --tgt-lang $tgt --models facebook/m2m100_418M \
        facebook/nllb-200-distilled-600M facebook/mbart-large-50-many-to-many-mmt Helsinki-NLP/opus-mt-$src-$tgt --error $error --batch-size 32 --max-length 1024
    done
done

