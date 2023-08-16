from datasets import load_from_disk
from transformers import AutoTokenizer

# Load data
#raw_dataset = load_dataset("royal42/lichess_elite_games")
raw_dataset = load_from_disk("../../../data/dataset")

# Load pretrained tokenizer
#tokenizer = AutoTokenizer.from_pretrained("royal42/chess_tokenizer", use_fast=True)
tokenizer = AutoTokenizer.from_pretrained("../../../data/tokenizer", use_fast=True)

# Tokenize all of the data, this will take a bit unless its cached.
context_length = 256 # What does this do?

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

tokenized_datasets.save_to_disk('../../../data/tokenized_files')