# PUT IN LLAMA_FACTORY DICTIONARY
dataset_tuple=('YOUR_DATASET_NAME')
for dataset in "${dataset_tuple[@]}"
do
    base_model=llama-13b-hf
    template=vicuna
    num_train_epochs=6.0
    export CUDA_VISIABLE_DEVICES='0,1,2,3,4,5,6,7'
    NUMGPUS=$(echo $CUDA_VISIABLE_DEVICES | awk -F',' '{print NF}')
    TOTALBSZ=512
    BSZPERDEV=4
    GRADACC=$(($TOTALBSZ/$NUMGPUS/$BSZPERDEV))
    deepspeed --include=localhost:${CUDA_VISIABLE_DEVICES} --master_port 29503  ./src/train_bash.py \
            --deepspeed ./examples/deepspeed/ds_z3_offload_config.json \
            --stage sft \
            --do_train \
            --model_name_or_path ../pre_trained_model/$base_model \
            --dataset $dataset \
            --dataset_dir ./data \
            --template $template \
            --finetuning_type full \
            --output_dir ./checkpoints/$base_model/${dataset}_epoch_${num_train_epochs} \
            --overwrite_cache \
            --overwrite_output_dir \
            --cutoff_len 2048 \
            --preprocessing_num_workers 16 \
            --per_device_train_batch_size $BSZPERDEV \
            --per_device_eval_batch_size $BSZPERDEV \
            --gradient_accumulation_steps $GRADACC \
            --lr_scheduler_type cosine \
            --logging_steps 0.05 \
            --warmup_ratio 0.1 \
            --save_strategy 'no' \
            --learning_rate 2e-5 \
            --num_train_epochs $num_train_epochs \
            --ddp_timeout 1800000 \
            --fp16 \
            --plot_loss
done
