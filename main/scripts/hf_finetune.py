from datasets import load_from_disk
import transformers
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from transformers import GPT2LMHeadModel
from huggingface_hub import login
import os

def main():
    # Login to hf? 
    login()

    tokenizer = AutoTokenizer.from_pretrained("royal42/chess_tokenizer", use_fast=True) 

    # Load tokenized data
    tokenized_datasets = load_from_disk("./data/tokenized_files")
    
    # Create collator
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Split data into train and val
    split_dataset = tokenized_datasets["train"].train_test_split(
        test_size=.1, seed=42
    )

    # Setup model
    model_checkpoint = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_checkpoint)

    # Setup trainer classes
    training_args = TrainingArguments(
        output_dir="./gpt2chess_finetune",          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=32,   # batch size for evaluation
        evaluation_strategy="steps",
        eval_steps=5000,
        save_total_limit=5,
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        fp16=True,
        push_to_hub=True
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"]
    )

    trainer.train()

    trainer.save_model()

    trainer.push_to_hub()

if __name__ == "__main__":
    main()
