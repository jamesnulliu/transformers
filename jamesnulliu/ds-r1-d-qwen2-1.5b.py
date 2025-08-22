import torch

from transformers import AutoTokenizer, GenerationConfig, Qwen2ForCausalLM


MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def main():
    print(f"Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = Qwen2ForCausalLM.from_pretrained(MODEL_NAME)
    print(type(model))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model loaded on: {device}")

    prompt = ["I'm good, because", "Hello, I want to"]

    inputs = tokenizer(prompt, padding=True, return_tensors="pt").to(device)

    print("Generating text...")

    generation_config = GenerationConfig(
        max_length=20,
        num_beams=4,
        top_k=30,
        do_sample=True,
        early_stopping=True,
        soft_thinking=True,
        return_dict_in_generate=True,
        output_logits=True,
        output_scores=True,
        padding_side="left",
    )

    output = model.generate(inputs.input_ids, generation_config=generation_config)

    o1 = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
    o2 = tokenizer.decode(output.sequences[1], skip_special_tokens=True)

    print("\n--- Generated Output ---")
    print(o1)
    print("------------------------")
    print(o2)
    print("------------------------")


if __name__ == "__main__":
    main()
