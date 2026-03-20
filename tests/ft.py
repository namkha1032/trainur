
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from datasets import load_dataset

from transformers import ( 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    AutoTokenizer, 
    DataCollatorForLanguageModeling,
    Qwen3ForCausalLM )
import torch
import setproctitle
from peft import (
    LoraConfig,
    PeftModel
)


def tokenize(batch):
    return tokenizer(
        batch["horoscope"],
        truncation=True,
        max_length=512,
    )

if __name__ == "__main__":

    model_name = "Qwen/Qwen3-0.6B"
    model = Qwen3ForCausalLM.from_pretrained(model_name).to(dtype=torch.bfloat16, device="cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset("karthiksagarn/astro_horoscope", split="train")

    dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    dataset = dataset.train_test_split(test_size=0.1)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # set up lora
    lora_config = LoraConfig(
                r=4,
                target_modules= "all-linear",
                lora_alpha=8,
                lora_dropout=0.05
            )
    model = PeftModel(model, lora_config)
    model.print_trainable_parameters()


    training_args = TrainingArguments(
        output_dir="qwen3-finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=False,
        # WARNING: this is mixed precision training, change to normal later
        bf16_full_eval=True,
        learning_rate=2e-5,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    model.save_pretrained("")