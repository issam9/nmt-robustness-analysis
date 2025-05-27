# Set to fr to evaluate French probes
src=en

for tgt in "es" "de" "it" "nl"
do
    for error in "prep" "nounnum" "article" "morpheus"
    do
        for model in "opus" "mbart" "nllb" "m2m100"
        do
            python probing/evaluate_probes.py --data-path ./outputs/representations/encodings/$src-$tgt --models $model-$src-$tgt-$error \
            $model-$src-$tgt-clean-$error --errors $error --src $src --pooling last --probe-path outputs/probes/$src-$tgt
        done
    done
done 


for tgt in "es" "de" "it" "nl"
do
    for error in "prep" "nounnum" "article" "morpheus"
    do
        python probing/evaluate_probes.py --data-path ./outputs/representations/encodings/$src-$tgt --models opus-mt-$src-$tgt \
         nllb-200-distilled-600M mbart-large-50-many-to-many-mmt m2m100_418M --errors $error --src $src --pooling last --probe-path outputs/probes/$src-$tgt
        python probing/evaluate_probes.py --data-path ./outputs/representations/encodings/$error/$src-$tgt --models \
        nllb-200-distilled-600M m2m100_418M mbart-large-50-many-to-many-mmt opus-mt-$src-$tgt --errors $error --src $src --pooling last --probe-path outputs/probes/$error/$src-$tgt
    done
done
