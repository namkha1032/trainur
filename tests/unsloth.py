from fsspec.implementations.arrow import parse_qs
import torch
from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_unsloth():
    pass
    model_name = "unsloth/Qwen3-8B"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "checkpoints/unsloth/Qwen3-8B",
        # max_seq_length = 2048, # Choose any for long context!
        # load_in_4bit = True,  # 4 bit quantization to reduce memory
        # load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
        # full_finetuning = False, # [NEW!] We have full finetuning now!
        # local_files_only = True,
        use_exact_model_name = True
        # token = "YOUR_HF_TOKEN", # HF Token for gated models
    )
    model.to(device="cuda:1")
    # model.save_pretrained(f'checkpoints/{model_name}')
    # tokenizer.save_pretrained(f'checkpoints/{model_name}')
    pass

def load_trans():
    model_name = "checkpoints/unsloth/Qwen3-8B"

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name
    ).to(device="cuda:3")
    # model.save_pretrained(f'checkpoints/{model_name}')
    # tokenizer.save_pretrained(f'checkpoints/{model_name}')
    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("thinking content:", thinking_content)
    print("content:", content)




if __name__ == "__main__":
    load_trans()