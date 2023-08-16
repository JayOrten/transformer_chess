# Run this on an edge node to get necessary modules from HF

from datasets import load_dataset
from transformers import AutoTokenizer

# Load data
raw_dataset = load_dataset("royal42/lichess_elite_games")

# Save to disk
raw_dataset.save_to_disk("../../../data/dataset")

# Load pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained("royal42/chess_tokenizer", use_fast=True)

# Save to disk
tokenizer.save_pretrained("../../../data/tokenizer")