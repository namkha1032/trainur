import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer

# ==============================================================================
# 1. Configuration
# ==============================================================================
model_id = "meta-llama/Llama-2-7b-chat-hf" # Replace with your target model
dataset_name = "timdettmers/openassistant-guanaco" # Replace with your dataset
output_dir = "./results"

# ==============================================================================
# 2. Load and Prepare Dataset
# ==============================================================================
# The 'timdettmers/openassistant-guanaco' dataset already has a 'text' column 
# formatted for conversational training.
dataset = load_dataset(dataset_name, split="train")

# ==============================================================================
# 3. Quantization Configuration (QLoRA)
# ==============================================================================
# Best practice: 4-bit NormalFloat (NF4) quantization with bfloat16 compute dtype
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ==============================================================================
# 4. Load Model and Tokenizer
# ==============================================================================
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# Llama 2 doesn't have a default pad token, so we use the EOS token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Best practice for causal LM training

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Enable gradient checkpointing to save VRAM
model.gradient_checkpointing_enable()

# Prepare the model for k-bit training (freezes base weights, casts layernorms to fp32)
model = prepare_model_for_kbit_training(model)

# ==============================================================================
# 5. LoRA Configuration
# ==============================================================================
# Best practice: Target ALL linear layers (q_proj, k_proj, v_proj, o_proj, etc.)
# rather than just attention heads. This yields performance closer to full fine-tuning.
peft_config = LoraConfig(
    r=16, # Rank of the adapter
    lora_alpha=32, # Alpha scaling
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)

# ==============================================================================
# 6. Training Arguments
# ==============================================================================
training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4, # Increase this if you run out of memory
    optim="paged_adamw_32bit",     # Uses paged optimizer to prevent memory spikes
    logging_steps=10,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,                     # Use bfloat16 if your GPU supports it (Ampere or newer)
    max_grad_norm=0.3,             # Gradient clipping
    max_steps=100,                 # Change to num_train_epochs for a full run
    warmup_ratio=0.03,
    group_by_length=True,          # Speeds up training by grouping sequences of similar length
    lr_scheduler_type="constant",
    save_strategy="steps",
    save_steps=25,
)

# ==============================================================================
# 7. Initialize SFTTrainer and Train
# ==============================================================================
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text", # The column in your dataset containing the formatted text
    max_seq_length=512,        # Truncate sequences to save memory
    tokenizer=tokenizer,
    args=training_arguments,
)

# Train the model
print("Starting training...")
trainer.train()

# ==============================================================================
# 8. Save the Fine-Tuned Adapters
# ==============================================================================
trainer.model.save_pretrained(f"{output_dir}/final_adapter")
tokenizer.save_pretrained(f"{output_dir}/final_adapter")
print("Training complete and adapter saved!")