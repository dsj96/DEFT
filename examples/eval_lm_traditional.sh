dataset_tuple=('YOUR_DATASET_NAME')
base_model=llama2-13b-hf
tasks_tuple=('arc_challenge' 'truthfulqa_mc2' 'mmlu' 'hellaswag')
num_fewshot_tuple=(25 0 5 10)
for dataset in "${dataset_tuple[@]}"
do
    for index in "${!tasks_tuple[@]}"
    do
        tasks="${tasks_tuple[$index]}"
        num_fewshot="${num_fewshot_tuple[$index]}"
        lm_eval --model hf \
            --model_args pretrained=../checkpoints/$base_model/$dataset,revision=step100000,dtype="float16" \
            --tasks $tasks \
            --num_fewshot $num_fewshot \
            --device cuda:1 \
            --output_path ./out/$tasks/$base_model/$dataset \
            --batch_size 1 \
            --cache_requests refresh
    done
done
