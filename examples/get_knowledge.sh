export CUDA_VISIABLE_DEVICES='0'

deepspeed --include=localhost:${CUDA_VISIABLE_DEVICES} --master_port 29700  src/detect_ds3.py \
	--model_path '../pre_trained_model/llama2-13b-hf' \
	--start_index 0 \
	--end_index 320000
