from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import GPT2LMHeadModel, AutoConfig
from transformers import Trainer, TrainingArguments
from huggingface_hub import login

def main():
    # Login to hf
    login()

    # Load data
    raw_dataset = load_dataset("royal42/lichess_elite_games")

    # Load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained("royal42/chess_tokenizer")

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

    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_ctx=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    model = GPT2LMHeadModel(config)

    args = TrainingArguments(
        output_dir="./gpt2chess_scratch",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="steps",
        eval_steps=5_000,
        logging_steps=5_000,
        gradient_accumulation_steps=8,
        num_train_epochs=10,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=5_000,
        fp16=True,
        push_to_hub=True
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
    )

    trainer.train()

    trainer.save_model()

    trainer.push_to_hub()

if __name__ == "__main__":
    main()
 