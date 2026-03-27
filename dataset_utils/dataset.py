from multiprocessing.spawn import prepare
import os
import torch
from datasets import load_dataset, Value
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase, default_data_collator

def exists(x):
    return x is not None

def get_dataset(dataset_name, path=None):
    if dataset_name == 'All_seqs':
        data_file = path
        dataset = load_dataset("csv", data_files=[data_file])
        # alldata
        train_ds = dataset['train']
        
        train_test_ds = train_ds.train_test_split(test_size=0.1, seed=42)
        train_ds = train_test_ds['train']
        
        test_ds = train_test_ds['test']
        test_val_ds = test_ds.train_test_split(test_size=0.5, seed=42)
        
        dataset['train'] = train_ds
        dataset['test'] = test_val_ds['train']
        dataset['valid'] =  test_val_ds['test']
    elif dataset_name == 'cls_seqs':
        data_file = path
        dataset = load_dataset("csv", data_files=[data_file])
        # alldata
        train_ds = dataset['train']
        
        train_test_ds = train_ds.train_test_split(test_size=0.2, seed=42)
        train_ds = train_test_ds['train']
        
        test_ds = train_test_ds['test']
        test_val_ds = test_ds.train_test_split(test_size=0.5, seed=42)
        
        dataset['train'] = train_ds
        dataset['test'] = test_val_ds['train']
        dataset['valid'] =  test_val_ds['test']
    elif dataset_name == 'csv' and path:
        dataset = load_dataset("csv", data_files=[path])
        # Default split for generic csv if only one file provided
        if 'train' in dataset:
            ds = dataset['train'].train_test_split(test_size=0.1, seed=42)
            dataset['train'] = ds['train']
            test_val = ds['test'].train_test_split(test_size=0.5, seed=42)
            dataset['test'] = test_val['train']
            dataset['valid'] = test_val['test']
    else:
        raise ValueError(f"Dataset {dataset_name} with path {path} not supported. Please use 'All_seqs', 'cls_seqs' or 'csv' with a valid --data_path.")
        
    return dataset

def collate_fn(batch):
    batch_dt = {}
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    batch_cdr3 = []
    for dic in batch:
        batch_input_ids.append(dic['input_ids'])
        batch_attention_mask.append(dic['attention_mask'])
        batch_cdr3.append(dic['input_ids'][98:108]+[1])

    batch_dt['input_ids'] = torch.tensor(batch_input_ids,dtype=torch.long)
    batch_dt['attention_mask'] = torch.tensor(batch_attention_mask,dtype=torch.long)
    batch_dt['cdr3']  = torch.tensor(batch_cdr3,dtype=torch.long)

    return batch_dt
def get_dataloader(args, dataset, model_config, tokenizer, max_seq_len):
    def tokenization(example):
        return tokenizer(example["AASeq"])

    dataset = dataset.map(tokenization,remove_columns='AASeq')
    return DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=args.train_batch_size,
            shuffle=True,
            pin_memory = True
        )
