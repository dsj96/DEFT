# INFERENCE
# FastChat/fastchat/llm_judge
dataset_tuple=('YOUR_DATASET_NAME')
for dataset in "${dataset_tuple[@]}"
do
    base_model=llama2-13b-hf
    DEVICES="0,1,2,3,4,5,6,7"
    export CUDA_VISIBLE_DEVICES=$DEVICES
    NUMGPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
    CUDA_VISIBLE_DEVICES=$DEVICES python gen_model_answer.py --model-path ../../../checkpoints/$base_model/${dataset}  --model-id vicuna_${base_model}_${dataset} --num-gpus-per-model 1 --num-gpus-total $NUMGPUS
    CUDA_VISIBLE_DEVICES=$DEVICES python gen_model_answer.py --model-path ../../../checkpoints/$base_model/${dataset}  --model-id vicuna_${base_model}_${dataset} --num-gpus-per-model 1 --num-gpus-total $NUMGPUS --bench-name alpaca_eval
done

# EVALUATION
# FastChat/fastchat/llm_judge
export OPENAI_API_KEY=YOUR_OPENAI_KEY
base_model=llama2-13b-hf
dataset='YOUR_DATASET_NAME'
model_id=vicuna_${base_model}_${dataset}

python3  gen_judgment.py  --model-list $model_id --parallel 1 --judge-model gpt-4 --bench-name mt_bench
cp ./data/mt_bench/model_judgment/gpt-4_single_${model_id}.jsonl ./data/mt_bench/model_judgment/gpt-4_single_${model_id}_tmp.jsonl
mv ./data/mt_bench/model_judgment/gpt-4_single_${model_id}_tmp.jsonl  ./data/mt_bench/model_judgment/gpt-4_single.jsonl
python show_result.py --model-list ${model_id}
