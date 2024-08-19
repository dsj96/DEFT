knowledge_model_tuple=("llama-13b-hf" "llama2-13b-hf" "Mistral-7B-v0.1")
stages_tuple=("_stage_c4q3_deberta-v3-base") #("stage_1" "stage_2" "stage_3" "stage_4" "stage_5")
knowledge=20 # TODO:
rank_model=deberta-v3-base
for knowledge_model in "${knowledge_model_tuple[@]}"
do
	for stage in "${stages_tuple[@]}"
	do
		echo knowledge_model=$knowledge_model
		echo stage=$stage
		GPUIDX="3"
		DATASIZE=10000
		THETA=0.9
		emb_base_model=sentence-transformers-e5-large-v2
		
		DATAPATH="../data/deita_sota_pool/model_metric/${stage}/qck_${knowledge_model}_deita_sota_pool_305263_token_4096.json"
		OTHERDATA=../data/deita_sota_pool/emb/${emb_base_model}/emb.pkl    # PATH/TO/EMBEDDING_FILE
		OUTPUTPATH=../data/deita_sota_pool/model_metric/${stage}/_qck_stage_c4q3_${emb_base_model}_${knowledge_model}_${rank_model}_deita_sota_pool_305263_${DATASIZE}_${THETA}_${knowledge}.json   # PATH/TO/OUTPUTS

		CUDA_VISIBLE_DEVICES=$GPUIDX python examples/pipelines/combined_filter.py \
				--data_path $DATAPATH \
				--other_data_path $OTHERDATA \
				--output_path $OUTPUTPATH \
				--threshold $THETA \
				--data_size $DATASIZE \
				--device 0
	done
done
