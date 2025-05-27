# Set to fr for French probing
src=en

for tgt in "es" "de" "nl" "it"
do
    for error in "prep" "article" "nounnum" "morpheus"
    do
        for model in "opus" "mbart" "nllb" "m2m100"
        do
            python ./probing/train_all_probes.py --models $model-$src-$tgt-$error $model-$src-$tgt-clean-$error --data-path ./outputs/representations/encodings/$src-$tgt --save-path ./outputs/probes/$src-$tgt --src $src --errors $error --pooling last
        done
    done
done


for lang in "es" "de" "nl" "it"
do
    for error in "prep" "article" "nounnum" "morpheus"
    do
        python ./probing/train_all_probes.py --models opus-mt-$src-$tgt nllb-200-distilled-600M mbart-large-50-many-to-many-mmt m2m100_418M --data-path ./outputs/representations/encodings/$src-$tgt --save-path ./outputs/probes/$src-$tgt --src $src --errors $error --pooling last
    done
done
