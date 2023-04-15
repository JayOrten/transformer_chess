from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from huggingface_hub import notebook_login
from transformers import GPT2LMHeadModel

def main():
    # Login to hf
    notebook_login()

    # Load data
    raw_dataset = load_dataset("royal42/lichess_elite_games")

    # Load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained("../data/tokenizations/chess-tokenizer/") 

    # Tokenize all of the data, this will take a bit unless its cached.
    context_length = 256

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
        num_train_epochs=10,              # total number of training epochs
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=32,   # batch size for evaluation
        evaluation_strategy="steps",
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

if __name__ == " __main__":
    main()