from transformers import AutoModel, AutoProcessor, LlavaForConditionalGeneration
import torch

def main():
    pass
    model_id = "llava-hf/llava-1.5-7b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id
    ).to(dtype=torch.bfloat16, device="cuda:1")
    processor = AutoProcessor.from_pretrained(model_id)
    pass


if __name__ == "__main__":
    main()