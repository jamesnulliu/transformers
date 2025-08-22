import torch

from transformers import AutoTokenizer, GenerationConfig, OPTForCausalLM


model_name = "facebook/opt-125m" 
print(f"Loading model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = OPTForCausalLM.from_pretrained(model_name)
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
    padding_side='left',
)

output = model.generate(inputs.input_ids, generation_config=generation_config)

o1 = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
o2 = tokenizer.decode(output.sequences[1], skip_special_tokens=True)

print("\n--- Generated Output ---")
print(o1)
print("------------------------")
print(o2)
print("------------------------")

