# -*- coding: utf-8 -*-
import json
import random
import os
import time
from openai_access import call_chatgpt
from depth import createConstraintsPrompt, createDeepenPrompt, createConcretizingPrompt, createReasoningPrompt
from breadth import createBreadthPrompt
import pdb
from tqdm import tqdm


file_path = './Evol/evol_instruction/alpaca_data_evol_3.json'
epoch=4
output_file_path = f'alpaca_data_evol_{epoch}.json'
processed_dir = "./processed_data/"
num_samples_to_evol = 2000
seed = 44 # evol1_seed=42 epoch=1, evol2_seed=42 epoch=2, evol3_seed=43 epoch=3, evol4_seed=44 epoch=4
fr = open(file_path,'r')
all_objs = json.load(fr)
evol_objs = []


import pickle
def read_pkl(fname):
    with open(fname, 'rb') as fo:
        pkl_data = pickle.load(fo, encoding='bytes')
    return pkl_data

def write_pkl(pkl_data, fname=processed_dir+ os.path.basename(output_file_path).split('.')[0]+".pickle"):
    fo = open(fname, 'wb')
    pickle.dump(pkl_data, fo)
    print("pkl_file write over!")

def setup_seed(seed=42):
    import torch
    import numpy as np
    import random
    from transformers import set_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    set_seed(seed)
def get_list(data_length, list_length):
    random_list = random.sample(range(data_length), list_length)
    return random_list
setup_seed(seed=seed) # evol1_seed=42 evol2_seed=42 evol3_seed=43 evol4_seed=44
if len(all_objs) > num_samples_to_evol:
    random_list = get_list(len(all_objs), list_length=num_samples_to_evol)
#     fin = open('select_index.txt', 'w', encoding='utf-8')
#     for item in random_list:
#         fin.write(str(item)+'\n')
#     fin.close()
    all_objs = [all_objs[i] for i in random_list]

    add_index_all_objs = []
    for index,item in enumerate(all_objs):
        item['index'] = random_list[index]
        add_index_all_objs.append(item)
    all_objs = add_index_all_objs

    with open(processed_dir+'add_index_all_objs.json', 'w') as file:
        json.dump(all_objs, file)
else:
    pass
pdb.set_trace()

for index,cur_obj in tqdm(enumerate(all_objs)):
    try:
        instruction = cur_obj['instruction'].strip() + '\r\n'+ cur_obj['input'].strip()
    except:
        print("No input, just instruction!")
        instruction = cur_obj['instruction'].strip()
    evol_prompts = []
    evol_prompts.append(createConstraintsPrompt(instruction))
    evol_prompts.append(createDeepenPrompt(instruction))
    evol_prompts.append(createConcretizingPrompt(instruction))
    evol_prompts.append(createReasoningPrompt(instruction))
    evol_prompts.append(createBreadthPrompt(instruction))
    type_list = ["createConstraintsPrompt", "createDeepenPrompt", "createConcretizingPrompt", "createReasoningPrompt", "createBreadthPrompt"]
    # selected_evol_prompt = random.choice(evol_prompts)
    idx = random.randint(0, len(evol_prompts) - 1)
    selected_evol_prompt = evol_prompts[idx]

    evol_instruction = call_chatgpt(selected_evol_prompt)
    
    if "#Rewritten Prompt#:" in evol_instruction:
        start_index = evol_instruction.find("#Rewritten Prompt#:") + len("#Rewritten Prompt#:")
        evol_instruction = evol_instruction[start_index:].strip(' \n')
    elif "Rewritten Prompt:" in evol_instruction:
        start_index = evol_instruction.find("Rewritten Prompt:") + len("Rewritten Prompt:")
        evol_instruction = evol_instruction[start_index:].strip(' \n')
    else:
        pass

    answer = call_chatgpt(evol_instruction)
    try:
        evol_objs.append({"instruction":evol_instruction,"output":answer, "index":cur_obj["index"], "type":type_list[idx]})
    except:
        time.sleep(10)
        try:
            evol_objs.append({"instruction":evol_instruction,"output":answer, "index":cur_obj["index"], "type":type_list[idx]})
        except:
            write_pkl(evol_objs)

with open(output_file_path, 'w') as f:
    json.dump(evol_objs, f, indent=4)
