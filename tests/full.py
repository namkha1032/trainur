import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Import required libraries for fine-tuning
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch
from peft import LoraConfig

if __name__=="__main__":
    # Initialize Weights & Biases (optional)
    # wandb.init(project="smollm3-finetuning")

    # Load SmolLM3 base model for fine-tuning
    model_name = "HuggingFaceTB/SmolLM3-3B-Base"
    new_model_name = "SmolLM3-Custom-SFT"

    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    instruct_model_name = "HuggingFaceTB/SmolLM3-3B"
    instruct_tokenizer = AutoTokenizer.from_pretrained(instruct_model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token
    tokenizer.padding_side = "right"  # Padding on the right for generation

    print(f"Model loaded! Parameters: {model.num_parameters():,}")

    print("=== PROCESSING GSM8K DATASET ===\n")

    gsm8k = load_dataset("openai/gsm8k", "main", split="train")  # Small subset for demo
    print(f"Original GSM8K example: {gsm8k[0]}")

    # Convert to chat format
    def process_gsm8k(examples):
        processed = []
        for question, answer in zip(examples["question"], examples["answer"]):
            messages = [
                {"role": "system", "content": "You are a math tutor. Solve problems step by step."},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
            processed.append(messages)
        return {"messages": processed}

    gsm8k_processed = gsm8k.map(process_gsm8k, batched=True, remove_columns=gsm8k.column_names)
    
    # Function to apply chat templates to processed datasets
    def apply_chat_template_to_dataset(dataset, tokenizer):
        """Apply chat template to dataset for training"""
        
        def format_messages(examples):
            formatted_texts = []
            
            for messages in examples["messages"]:
                # Apply chat template
                formatted_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False  # We want the complete conversation
                )
                formatted_texts.append(formatted_text)
            
            return {"text": formatted_texts}
        
        return dataset.map(format_messages, batched=True)

    # Apply to our processed GSM8K dataset
    gsm8k_formatted = apply_chat_template_to_dataset(gsm8k_processed, instruct_tokenizer)


    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear"
    )

    # Configure training parameters
    training_config = SFTConfig(
        # Model and data
        output_dir=f"./{new_model_name}",
        dataset_text_field="text",
        max_length=2048,
        
        # Training hyperparameters
        per_device_train_batch_size=2,  # Adjust based on your GPU memory
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        num_train_epochs=1,  # Start with 1 epoch
        max_steps=500,  # Limit steps for demo
        
        # Optimization
        warmup_steps=50,
        weight_decay=0.01,
        optim="adamw_torch",
        
        # Logging and saving
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        save_total_limit=2,
        
        # Memory optimization
        dataloader_num_workers=0,
        # group_by_length=True,  # Group similar length sequences
        
        # Hugging Face Hub integration
        # push_to_hub=False,  # Set to True to upload to Hub
        # hub_model_id=f"your-username/{new_model_name}",
        
        # Experiment tracking
        # report_to=["trackio"],  # Use trackio for experiment tracking
        # run_name=f"{new_model_name}-training",
    )

    print("Training configuration set!")
    print(f"Effective batch size: {training_config.per_device_train_batch_size * training_config.gradient_accumulation_steps}")

    lora_trainer = SFTTrainer(
        model=model,
        train_dataset=gsm8k_formatted ,  # dataset with a  "text" field or messages + dataset_text_field in config
        args=training_config,
        peft_config=peft_config,  # << enable LoRA
    )

    print("Starting LoRA training…")
    lora_trainer.train()
    pass