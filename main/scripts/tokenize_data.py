from datasets import load_dataset
from transformers import AutoTokenizer

# Load data
raw_dataset = load_dataset("royal42/lichess_elite_games")

# Load pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained("royal42/chess_tokenizer", use_fast=True) 

# Tokenize all of the data, this will take a bit unless its cached.
context_length = 5

def tokenize(element):
    outputs = tokenizer(
        element["text"],
        max_length=context_length,
        truncation=True
    )
    return outputs


tokenized_datasets = raw_dataset.map(
    tokenize, batched=True, remove_columns=raw_dataset["train"].column_names
)

tokenized_datasets.save_to_disk('./data/tokenized_files')