#!/bin/bash

declare -A lang_codes_nllb=( ["nl"]="nld_Latn" ["de"]="deu_Latn" ["it"]="ita_Latn" ["es"]="spa_Latn" ["en"]="eng_Latn" ["fr"]="fra_Latn" )
declare -A lang_codes_mbart=( ["nl"]="nl_XX" ["de"]="de_DE" ["it"]="it_IT" ["es"]="es_XX" ["en"]="en_XX" ["fr"]="fr_XX" )
declare -A lang_codes_m2m100=( ["nl"]="__nl__" ["de"]="__de__" ["it"]="__it__" ["es"]="__es__" ["en"]="__en__" ["fr"]="__fr__" )


models=(
"facebook/nllb-200-distilled-600M nllb lang_codes_nllb"
"facebook/mbart-large-50-many-to-many-mmt mbart lang_codes_mbart"
"facebook/m2m100_418M m2m100 lang_codes_m2m100"
"Helsinki-NLP/opus-mt-\$src-\$lang opus none"
)

errors=("prep" "article" "nounnum" "morpheus")

src=en

for lang in "es" "de" "it" "nl"
do
    for model in "${models[@]}"
    do
        IFS=' ' read -r model_name model_prefix lang_code_key <<< "$model"
        
        echo "Processing model: $model_name, prefix: $model_prefix, lang_code_key: $lang_code_key"
        
        if [ "$lang_code_key" != "none" ]; then
            lang_codes_var="${lang_code_key}[${src}]"
            src_code="${!lang_codes_var}"
            lang_codes_var="${lang_code_key}[${lang}]"
            lang_code="${!lang_codes_var}"
        else
            src_code=$src
            lang_code=$lang
        fi
        
        echo "Source: ${src_code}, Target: ${lang_code}"
        
        if [ -z "${lang_code}" ]; then
            echo "Error: Forced BOS token for target language '$lang' is not set."
            continue
        fi
        for error in "${errors[@]}"
        do
            if [$error=="morpheus"]; then
                curr_error=${model_prefix}
            python ./finetuning/run_translation.py \
            --model_name_or_path $model_name \
            --num_beams 5 \
            --do_train \
            --do_eval \
            --source_lang "${src_code}" \
            --target_lang "${lang_code}" \
            --data_dir ./data/finetuning-subset/$src/$lang \
            --output_dir ./outputs/models/${model_prefix}-$src-$lang-$error \
            --per_device_train_batch_size 16 \
            --gradient_accumulation_steps 4 \
            --generation_num_beams 5 \
            --per_device_eval_batch_size 8 \
            --overwrite_output_dir \
            --error $curr_error \
            --freeze_encoder False \
            --freeze_decoder True \
            --unfreeze_encoder_attn False \
            --save_total_limit 1 \
            --predict_with_generate \
            --max_steps 5000 \
            --report_to wandb \
            --run_name ${model_prefix}-$src-$lang-$error \
            --load_best_model_at_end True \
            --evaluation_strategy steps \
            --eval_steps 1000 \
            --save_steps 1000 \
            --forced_bos_token "${lang_code}" \
            --do_predict \
            --metric_for_best_model bleu \
            --greater_is_better True

            python ./finetuning/run_translation.py \
            --model_name_or_path $model_name \
            --num_beams 5 \
            --do_train \
            --do_eval \
            --source_lang "${src_code}" \
            --target_lang "${lang_code}" \
            --data_dir ./data/finetuning-subset/$src/$lang \
            --output_dir ./outputs/models/${model_prefix}-$src-$lang-clean-$error \
            --per_device_train_batch_size 16 \
            --gradient_accumulation_steps 4 \
            --generation_num_beams 5 \
            --per_device_eval_batch_size 8 \
            --overwrite_output_dir \
            --error $curr_error \
            --freeze_encoder False \
            --freeze_decoder True \
            --unfreeze_encoder_attn False \
            --use_clean True \
            --save_total_limit 1 \
            --predict_with_generate \
            --max_steps 5000 \
            --report_to wandb \
            --run_name ${model_prefix}-$src-$lang-clean-$error \
            --load_best_model_at_end True \
            --evaluation_strategy steps \
            --eval_steps 1000 \
            --save_steps 1000 \
            --forced_bos_token "${lang_code}" \
            --do_predict \
            --metric_for_best_model bleu \
            --greater_is_better True
        done
    done
done


# Set source language to French
src=fr

for lang in "es"
do
    for model in "${models[@]}"
    do
        IFS=' ' read -r model_name model_prefix lang_code_key <<< "$model"
        
        echo "Processing model: $model_name, prefix: $model_prefix, lang_code_key: $lang_code_key"
        
        if [ "$lang_code_key" != "none" ]; then
            lang_codes_var="${lang_code_key}[${src}]"
            src_code="${!lang_codes_var}"
            lang_codes_var="${lang_code_key}[${lang}]"
            lang_code="${!lang_codes_var}"
        else
            src_code=$src
            lang_code=$lang
        fi
        
        echo "Source: ${src_code}, Target: ${lang_code}"
        
        if [ -z "${lang_code}" ]; then
            echo "Error: Forced BOS token for target language '$lang' is not set."
            continue
        fi
        
        for error in "${errors[@]}"
        do
            if [$error=="morpheus"]; then
                curr_error=${model_prefix}
            python ./finetuning/run_translation.py \
            --model_name_or_path $model_name \
            --num_beams 5 \
            --do_train \
            --do_eval \
            --source_lang "${src_code}" \
            --target_lang "${lang_code}" \
            --data_dir ./data/finetuning-subset/$src/$lang \
            --output_dir ./outputs/models/${model_prefix}-$src-$lang-$error \
            --per_device_train_batch_size 16 \
            --gradient_accumulation_steps 4 \
            --generation_num_beams 5 \
            --per_device_eval_batch_size 8 \
            --overwrite_output_dir \
            --error $curr_error \
            --freeze_encoder False \
            --freeze_decoder True \
            --unfreeze_encoder_attn False \
            --save_total_limit 1 \
            --predict_with_generate \
            --max_steps 5000 \
            --report_to wandb \
            --run_name ${model_prefix}-$src-$lang-$error \
            --load_best_model_at_end True \
            --evaluation_strategy steps \
            --eval_steps 1000 \
            --save_steps 1000 \
            --forced_bos_token "${lang_code}" \
            --do_predict \
            --metric_for_best_model bleu \
            --greater_is_better True

            python ./finetuning/run_translation.py \
            --model_name_or_path $model_name \
            --num_beams 5 \
            --do_train \
            --do_eval \
            --source_lang "${src_code}" \
            --target_lang "${lang_code}" \
            --data_dir ./data/finetuning-subset/$src/$lang \
            --output_dir ./outputs/models/${model_prefix}-$src-$lang-clean-$error \
            --per_device_train_batch_size 16 \
            --gradient_accumulation_steps 4 \
            --generation_num_beams 5 \
            --per_device_eval_batch_size 8 \
            --overwrite_output_dir \
            --error $curr_error \
            --freeze_encoder False \
            --freeze_decoder True \
            --unfreeze_encoder_attn False \
            --use_clean True \
            --save_total_limit 1 \
            --predict_with_generate \
            --max_steps 5000 \
            --report_to wandb \
            --run_name ${model_prefix}-$src-$lang-clean-$error \
            --load_best_model_at_end True \
            --evaluation_strategy steps \
            --eval_steps 1000 \
            --save_steps 1000 \
            --forced_bos_token "${lang_code}" \
            --do_predict \
            --metric_for_best_model bleu \
            --greater_is_better True
        done
    done
done