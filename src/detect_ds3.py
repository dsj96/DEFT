# https://huggingface.co/docs/transformers/deepspeed?models=pretrained+model#non-trainer-deepspeed-integration
# This script demonstrates how to use Deepspeed ZeRO in an inference mode when one can't fit a model
# into a single GPU
#
# 1. Use 1 GPU with CPU offload
# 2. Or use multiple GPUs instead
#
# First you need to install deepspeed: pip install deepspeed
#
# Here we use a 3B "bigscience/T0_3B" model which needs about 15GB GPU RAM - so 1 largish or 2
# small GPUs can handle it. or 1 small GPU and a lot of CPU memory.
#
# To use a larger model like "bigscience/T0" which needs about 50GB, unless you have an 80GB GPU -
# you will need 2-4 gpus. And then you can adapt the script to handle more gpus if you want to
# process multiple inputs at once.
#
# The provided deepspeed config also activates CPU memory offloading, so chances are that if you
# have a lot of available CPU memory and you don't mind a slowdown you should be able to load a
# model that doesn't normally fit into a single GPU. If you have enough GPU memory the program will
# run faster if you don't want offload to CPU - so disable that section then.
#
# To deploy on 1 gpu:
#
# deepspeed --num_gpus 1 t0.py
# or:
# python -m torch.distributed.run --nproc_per_node=1 t0.py
#
# To deploy on 2 gpus:
#
# deepspeed --num_gpus 2 t0.py
# or:
# python -m torch.distributed.run --nproc_per_node=2 t0.py

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.integrations import HfDeepSpeedConfig
import deepspeed
import os
import torch
from torch.cuda.amp import autocast
import pdb
import json
from tqdm import tqdm
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers

# distributed setup
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
torch.cuda.set_device(local_rank)
deepspeed.init_distributed()
import argparse

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default='../pre_trained_model/llama2-13b-hf')
    
    parser.add_argument('--data_path', type=str, default="../data/deita_sota_pool/deita_sota_pool_305263_token_4096.json")

    parser.add_argument('--start_index', type=int, default=0, help='')
    
    parser.add_argument('--end_index', type=int, default=100, help='')

    parser.add_argument('--max_length', type=int, default=2048, help='')

    parser.add_argument('--output_dir', type=str, default="../data/deita_sota_pool")

    args, _ = parser.parse_known_args()
    # args.cuda = not args.cuda and torch.cuda.is_available()
    return args

args = get_args()
print(args)
model_name = args.model_path
model_path = args.model_path
data_path = args.data_path
start_index = args.start_index
end_index = args.end_index
max_length = args.max_length
output_dir = args.output_dir


output_file = output_dir+ '/' + model_path.split('/')[-1] + f"_s{start_index}_e{end_index}" + f"_{data_path.split('/')[-1]}"


config = AutoConfig.from_pretrained(model_name)
model_hidden_size = config.hidden_size

# batch size has to be divisible by world_size, but can be bigger than world_size
train_batch_size = 1 * world_size

# ds_config notes
#
# - enable bf16 if you use Ampere or higher GPU - this will run in mixed precision and will be
# faster.
#
# - for older GPUs you can enable fp16, but it'll only work for non-bf16 pretrained models - e.g.
# all official t5 models are bf16-pretrained
#
# - set offload_param.device to "none" or completely remove the `offload_param` section if you don't
# - want CPU offload
#
# - if using `offload_param` you can manually finetune stage3_param_persistence_threshold to control
# - which params should remain on gpus - the larger the value the smaller the offload size
#
# For in-depth info on Deepspeed config see
# https://huggingface.co/docs/transformers/main/main_classes/deepspeed

# keeping the same format as json for consistency, except it uses lower case for true/false
# fmt: off
ds_config = {
    "fp16": {
        "enabled": True
    },
    "bf16": {
        "enabled": False
    },
    "zero_optimization": {
        "stage": 3,
        # "offload_param": {
        #     "device": "cpu",
        #     "pin_memory": True
        # },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": model_hidden_size * model_hidden_size,
        "stage3_prefetch_bucket_size": 0.9 * model_hidden_size * model_hidden_size,
        "stage3_param_persistence_threshold": 10 * model_hidden_size
    },
    "steps_per_print": 2000,
    "train_batch_size": train_batch_size,
    "train_micro_batch_size_per_gpu": 1,
    "wall_clock_breakdown": False
}
# fmt: on

# next line instructs transformers to partition the model directly over multiple gpus using
# deepspeed.zero.Init when model's `from_pretrained` method is called.
#
# **it has to be run before loading the model AutoModelForSeq2SeqLM.from_pretrained(model_name)**
#
# otherwise the model will first be loaded normally and only partitioned at forward time which is
# less efficient and when there is little CPU RAM may fail
dschf = HfDeepSpeedConfig(ds_config)  # keep this object alive

# now a model can be loaded.
model = AutoModelForCausalLM.from_pretrained(model_name)

# initialise Deepspeed ZeRO and store only the engine object
ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
ds_engine.module.eval()  # inference

# print(f'world_size:{world_size}')

# ds_engine = deepspeed.init_inference(model, 
#                                      config={
#                                            "dtype": torch.float16,
#                                             "tensor_parallel": {"tp_size": world_size },
#                                             "max_out_tokens" : 4096,
#                                         },
#                                      replace_with_kernel_inject=True,
#                                     )




rank = torch.distributed.get_rank()

tokenizer = AutoTokenizer.from_pretrained(model_name)

def calculatePerplexity(sentence, model, tokenizer, device):
    """
    exp(loss)
    """
    # input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = tokenizer.encode(sentence, return_tensors="pt", max_length=max_length, truncation=True) # add_special_tokens=False
    
    input_ids = input_ids.to(device)
    with torch.no_grad():
    # with autocast():
        outputs = model(input_ids, labels=input_ids)#, synced_gpus=True)
    loss, logits = outputs[:2]
    
    '''
    extract logits:
    '''
    # Apply softmax to the logits to get probabilities
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)

    # probabilities = torch.nn.functional.softmax(logits, dim=-1)
    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)
    return torch.exp(loss).item(), all_prob, loss.item(), outputs

def inference(model, tokenizer, sentence, device):
    pred = {}
    
    p1, all_prob, p1_likelihood, outputs = calculatePerplexity(sentence, model, tokenizer, device=device)
    pred["ppl"] = p1
    pred["loss"] = p1_likelihood
    # pred["outputs"] = outputs
    for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        k_length = int(len(all_prob)*ratio)
        if k_length ==0:
            k_length =1
        topk_prob = np.sort(all_prob)[:k_length]
        if np.isnan(np.mean(-np.mean(topk_prob))):
            print(f"topk_prob:{topk_prob} \t k_length:{k_length} \t ratio:{ratio}")
        pred[f"Min_{ratio*100}%_Prob"] = -np.mean(topk_prob).item()
    
    return pred


def _load_data(data_path: str) -> None:
    
    """
    Load data from data_path.
    
    data_path: str - path to json data file.
    """
    
    try:
        with open(data_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        with open(data_path, "r") as f:
            data = [json.loads(line) for line in f]
    
    return data


data = _load_data(data_path)[start_index:end_index]
model = ds_engine.module

pred_list = []
for index, item in tqdm(enumerate(data)):
    conversations = item['conversations']
    num_turns = int(len(conversations)/2)
    cur_pred = []
    for i in range(num_turns):
        cur_q = conversations[int(i*2)]['value']
        cur_a = conversations[int(i*2+1)]['value']
        cur_pred_q = inference(model, tokenizer, cur_q, device=local_rank)
        cur_pred_a = inference(model, tokenizer, cur_a, device=local_rank)
        cur_pred.append([cur_pred_q, cur_pred_a])
    pred_list.append(cur_pred)

with open(output_file, "w") as json_file:
    json.dump(pred_list, json_file)
