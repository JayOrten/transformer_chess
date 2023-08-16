# Run this to save model to HF
from datasets import load_from_disk
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import GPT2LMHeadModel, AutoConfig
from transformers import Trainer, TrainingArguments
from huggingface_hub import login

login()

model = GPT2LMHeadModel.from_pretrained('../../../data/models/gpt2chess_scratch_test816')

model.push_to_hub('chess-transformer-test-816')