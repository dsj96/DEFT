# -*- coding: utf-8 -*-
from sentence_transformers import SentenceTransformer
from datasets import Dataset, load_dataset, load_metric, load_from_disk
from typing import Sequence, Dict, List
import pandas as pd
from tqdm import tqdm


import os
import json
from functools import partial

from deita.selection.embedder.conversation import get_conv_template

sub_share_gpt_arrow = '../data/deita_sota_pool/deita_sota_pool_305263_token_4096_arrow'
conv_template = "vicuna_v1.1" # vicuna_v1.1
only_answer = False # False
max_length = 4096 # 2048
# device_ids = [0, 1]
model_path = '../pre_trained_model/sentence-transformers-e5-large-v2'
output_path = '../data/deita_sota_pool/emb' + '/' +  model_path.split('/')[-1]
print(output_path)

if not os.path.exists(output_path):
    os.makedirs(output_path)

try:
    
    model = SentenceTransformer(model_path).cuda()
    print("gpu")
except:
    
    model = SentenceTransformer(model_path)
    print("cpu")

raw_dataset = load_from_disk(sub_share_gpt_arrow)






def preprocess(
        samples: Dataset,
        conv_template="vicuna_v1.1",
        only_answer=False,
        max_length=2048
    ) -> Dict:
        
        conv = get_conv_template(conv_template) # conv=Conversation(name='vicuna_v1.1', system="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.", roles=('USER', 'ASSISTANT'), messages=[], offset=0, sep_style=<SeparatorStyle.ADD_COLON_TWO: 2>, sep=' ', sep2='</s>', stop_str=None, stop_token_ids=None)
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]} # {"human": 'USER', "gpt": 'ASSISTANT'}

        sources = samples["conversations"]
        sample_ids = samples["input_idx"]

        # Apply prompt templates
        conversations = []

        if not only_answer: # only_answer=False
            for i, source in enumerate(sources):
                if roles[source[0]["from"]] != conv.roles[0]:
                    # Skip the first one if it is not from human
                    source = source[1:]

                conv.messages = []
                for j, sentence in enumerate(source):
                    role = roles[sentence["from"]]
                    # assert role == conv.roles[j % 2], f"{i}"
                    # assert role == conv.roles[j % 2], breakpoint()
                    conv.append_message(role, sentence["value"])
                conversations.append(conv.get_prompt())
        else:
            for i, source in enumerate(sources):
                if roles[source[0]["from"]] != conv.roles[0]:
                    # Skip the first one if it is not from human
                    source = source[1:]

                messages = []
                for j, sentence in enumerate(source):
                    if j % 2 == 0:
                        continue
                    messages.append(sentence["value"])
                conversations.append("\n".join(messages))


        
        return dict(
            conversations=conversations,
            idx = sample_ids
        )

preprocess_func = partial(preprocess,
                        conv_template = conv_template, # vicuna_v1.1
                        only_answer = only_answer, # False
                        max_length = max_length # 2048
                         )


tokenized_datasets = raw_dataset.map(
    preprocess_func,
    batched = True,
    # num_proc = 1,
    remove_columns = ["conversations", "specific_length"],
    desc = "Tokenizing and reformatting instruction data"
)


df_dict = {
    'embedding':[],
    'idx':[]
}

for idx, item in enumerate(tqdm(tokenized_datasets)):
    embeddings = model.encode([item['conversations']]).tolist()
    df_dict['embedding'].append(embeddings)
    df_dict['idx'].append(idx)


df = pd.DataFrame(df_dict)
df.to_pickle(output_path + '/emb.pkl')
