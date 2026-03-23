import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import torch

if __name__=="__main__":
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    dataset = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations")

    # Configure model and tokenizer
    model_name = "HuggingFaceTB/SmolLM2-135M"
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name).to(
        device
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
    # Setup chat template
    # model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

    # Configure trainer
    training_args = SFTConfig(
        output_dir="./sft_output",
        max_steps=1000,
        per_device_train_batch_size=4,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=100,
        eval_strategy="steps",
        eval_steps=50,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
    )

    # Start training
    trainer.train()