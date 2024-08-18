import json
from torch.utils.data import Dataset
from accelerate import Accelerator
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, AdamW
import torch
from tqdm import tqdm
from rank_loss import RankLoss
import numpy as np
import os
import argparse
import tempfile
import copy
import random
import pdb
import re
import os
import gc
from torch.cuda.amp import autocast
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
torch.backends.cudnn.benchmark=True

class RerankData(Dataset):
    def __init__(self, data, tokenizer, label=True, max_length=2048):
        self.data = data
        self.tokenizer = tokenizer
        self.label = label
        self.max_length = max_length


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        item = self.data[item]
        neg_num = len(item['retrieved_passages'])
        query = item['query']

        passages = [str(psg['text']) for psg in item['retrieved_passages']][:neg_num]
        return query, passages

    def collate_fn(self, data):
        query, passages = zip(*data)
        query = sum(query, []) # tuple -> List
        passages = sum(passages, []) # # tuple -> List
        if self.label: # quality
            features = self.tokenizer(query, passages, padding=True, truncation=True, return_tensors="pt",
                                      max_length=self.max_length)
        else: # complexity
            features = self.tokenizer(query, padding=True, truncation=True, return_tensors="pt",
                                      max_length=self.max_length)
        return features # features['input_ids'].shape=torch.Size([20, 187])  features['attention_mask'].shape=torch.Size([20, 187])


def receive_response(data, responses):
    def clean_response(response: str):
        pattern = r'\[\d+\]'
        matches = re.findall(pattern, response)
        response = ' > '.join(matches)
        new_response = ''
        for c in response:
            if not c.isdigit():
                new_response += ' '
            else:
                new_response += c
        new_response = new_response.strip()
        return new_response

    def remove_duplicate(response):
        new_response = []
        for c in response:
            if c not in new_response:
                new_response.append(c)
        return new_response
    test_num = -1
    new_data = []
    for item, response in zip(data, responses): # reponse='[5] > [18] > [14] > [1] > [15] > [3] > [4] > [13] > [7] > [6] > [12] > [9] > [10] > [11] > [16] > [2] > [19] > [20] > [8] > [17]'
        test_num = test_num + 1

        response = clean_response(response) # '5     18     14     1     15     3     4     13     7     6     12     9     10     11     16     2     19     20     8     17
        response = [int(x) - 1 for x in response.split()] # [5, 18, 14...]
        response = remove_duplicate(response)
        passages = item['retrieved_passages']
        original_rank = [tt for tt in range(len(passages))] # [0, 1, .., len]
        response = [ss for ss in response if ss in original_rank]
        response = response + [tt for tt in original_rank if tt not in response] # [5, 18, 14...]

        new_passages = [passages[ii] for ii in response]
        new_querys   = [item['query'][ii] for ii in response]
        new_data.append({'query': new_querys,
                         'positive_passages': item['positive_passages'],
                         'retrieved_passages': new_passages})
    return new_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='microsoft/deberta-v3-base')
    parser.add_argument('--loss', type=str, default='rank_net')
    parser.add_argument('--data', type=str, default='data/marco-train-10k.jsonl')
    parser.add_argument('--save_path', type=str, default='out/deberta-rank_net')
    parser.add_argument('--permutation', type=str, default='marco-train-10k-gpt3.5.json')
    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_eval', type=bool, default=True)
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--ylabel', type=bool, default=False)
    args = parser.parse_args()

    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))
    return args


def train(args):
    print(args)
    model_name = args.model
    loss_type = args.loss
    data_path = args.data
    save_path = args.save_path
    permutation = args.permutation
    max_length = args.max_length
    
    accelerator = Accelerator()

    # Create cross encoder model
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = 1
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.unk_token
    model.config.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    data     = json.load(open(data_path))
    response = json.load(open(permutation))

    if 'complexity' in args.save_path:
        label = False
    if 'quality' in args.save_path:
        label = True

    data = receive_response(data, response)
    dataset = RerankData(data, tokenizer, label=label, max_length=max_length)

    # Prepare data loader
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn,
                                              batch_size=1, shuffle=True, num_workers=0)

    optimizer = AdamW(model.parameters(), 2e-5)
    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

    # Prepare loss function
    loss_function = getattr(RankLoss, loss_type)

    # Train for 3 epoch
    for epoch in range(3):
        accelerator.print(f'Training {save_path} {epoch}')
        accelerator.wait_for_everyone()
        model.train()
        tk0 = tqdm(data_loader, total=len(data_loader))
        loss_report = []
        for batch in tk0:
            with accelerator.accumulate(model):
                gc.collect()
                torch.cuda.empty_cache()
                # use_cache=False
                # with autocast():
                out = model(**batch) # batch[0]=Encoding(num_tokens=159, attributes=[ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing])
                logits = out.logits # torch.Size([20, 1])
                logits = logits.view(-1, batch['input_ids'].shape[0])
                # [[1 / (i + 1) for i in range(logits.size(1))]] = [[1.0, 0.5, 0.3333333333333333, 0.25, 0.2, 0.16666666666666666, 0.14285714285714285, 0.125, 0.1111111111111111, 0.1, 0.09090909090909091, 0.08333333333333333, 0.07692307692307693, 0.07142857142857142, 0.06666666666666667, 0.0625, 0.058823529411764705, 0.05555555555555555, 0.05263157894736842, 0.05]]
                if args.ylabel:
                    y_true = torch.tensor([[1 / (i + 1) for i in range(logits.size(1))]] * logits.size(0)).cuda()
                    loss = loss_function(logits, y_true) # logits=torch.Size([1, 20])   y_true=torch.Size([1, 20])
                else:
                    loss = loss_function(logits)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                optimizer.zero_grad()
                loss_report.append(accelerator.gather(loss).mean().item())
            tk0.set_postfix(loss=sum(loss_report) / len(loss_report))
        accelerator.wait_for_everyone()

    # Save model
    state_dict = accelerator.get_state_dict(model)
    unwrap_model = accelerator.unwrap_model(model)
    os.makedirs(save_path, exist_ok=True)
    unwrap_model.save_pretrained(save_path, state_dict=state_dict, safe_serialization=True)
    accelerator.save_state(save_path, safe_serialization=False)
    return model, tokenizer



if __name__ == '__main__':
    args = parse_args()
    model, tokenizer = None, None
    if args.do_train:
        model, tokenizer = train(args)
        tokenizer.save_pretrained(args.save_path)

