from datasets import load_dataset
from transformers import ( 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    AutoTokenizer, 
    DataCollatorForLanguageModeling,
    Qwen3ForCausalLM )
import torch
def tokenize(batch):
    return tokenizer(
        batch["horoscope"],
        truncation=True,
        max_length=512,
    )

if __name__ == "__main__":

    model_name = "Qwen/Qwen3-0.6B"
    model = Qwen3ForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset("karthiksagarn/astro_horoscope", split="train")

    dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    dataset = dataset.train_test_split(test_size=0.1)

    training_args = TrainingArguments(
        output_dir="qwen3-finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        bf16=True,
        learning_rate=2e-5,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )