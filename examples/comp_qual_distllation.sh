# refenrece: https://github.com/sunnweiwei/RankGPT/tree/main

# COMPLEXITY
do_train=true
max_length=1024
data=alpaca-train-complexity-2k
permutation=alpaca-train-complexity-rank-2k
model=deberta-v3-base
gpu_ids='0,1,2,3,4,5,6,7'
stage=_stage_0_deberta-v3-base
my_tuple=("rank_net") # "pointwise_rmse" "pointwise_bce" "list_net" "lambda_loss"
for loss in "${my_tuple[@]}"
do
    accelerate launch --num_processes 8 --gpu_ids $gpu_ids  --config_file ./deepspeed/default_config_8_fp32.yaml  comp_qual_distillation.py \
    --model ./pre_trained_model/$model \
    --loss $loss \
    --data  ./data/RankGPT/$stage/$data.json \
    --permutation ./data/RankGPT/$stage/$permutation.json \
    --save_path ./checkpoints/$model/$data/${stage}/$loss-$max_length-ylabel \
    --do_train $do_train \
    --max_length $max_length \
    --ylabel=True
done


# QUALITY
do_train=true
max_length=2048
data=alpaca-train-quality-2k
permutation=alpaca-train-quality-rank-2k
model=deberta-v3-base
gpu_ids='0,1,2,3,4,5,6,7'
stage=_stage_0_deberta-v3-base
main_process_port=29526
my_tuple=("rank_net") # "pointwise_rmse" "pointwise_bce" "list_net" "lambda_loss"
for loss in "${my_tuple[@]}"
do
    accelerate launch --num_processes 8 --gpu_ids $gpu_ids  --main_process_port $main_process_port --config_file ./deepspeed/default_config_8_fp32.yaml  comp_qual_distillation.py \
    --model ./pre_trained_model/$model \
    --loss $loss \
    --data  ./data/RankGPT/$stage/$data.json \
    --permutation ./data/RankGPT/$stage/$permutation.json \
    --save_path ./checkpoints/$model/$data/$stage/$loss-$max_length-ylabel \
    --do_train $do_train \
    --max_length $max_length \
    --ylabel=True
done
