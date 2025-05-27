# Set to fr for French encodings
src=en

# Encode with Noise-finetuned and Clean-finetuned models
for tgt in "es" "de" "it" "nl"
do
    for model in "m2m100" "mbart" "nllb" "opus"
    do
        for error in "prep" "nounnum" "article" "morpheus"
        do
            # Encode noisy sets
            python representations/encode.py --src-lang $src --tgt-lang $tgt --models outputs/models/$model-$src-$tgt-$error outputs/models/$model-$src-$tgt-clean-$error --error $error --pooling last --batch-size 32
            # Encode the clean test set for similarity analysis
            python representations/encode.py --src-lang $src --tgt-lang $tgt --models outputs/models/$model-$src-$tgt-$error outputs/models/$model-$src-$tgt-clean-$error --error $error --pooling last --batch-size 32 --encode-clean
        done
    done
done

# Encode with Base models
for tgt in "es" "de" "it" "nl"
do
    for error in "prep" "nounnum" "article" "morpheus"
    do
        #Encode noisy sets
        python representations/encode.py --src-lang $src --tgt-lang $tgt --models facebook/nllb-200-distilled-600M facebook/m2m100_418M facebook/mbart-large-50-many-to-many-mmt Helsinki-NLP/opus-mt-$src-$tgt --error $error --pooling last --batch-size 32
        # Encode the clean test set for similarity analysis
        python representations/encode.py --src-lang $src --tgt-lang $tgt --models facebook/nllb-200-distilled-600M facebook/m2m100_418M facebook/mbart-large-50-many-to-many-mmt Helsinki-NLP/opus-mt-$src-$tgt --error $error --pooling last --batch-size 32 --encode-clean
    done
done
