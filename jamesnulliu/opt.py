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

# print(logprobs.shape)

# from transformers.hack.vis import *  # noqa: F403


# setup_matplotlib_style()


# fig1, ax1 = visualize_logprob_line(
#     logprobs[0], title=f"Log Probability Distribution "
# )
# # Save
# fig1.savefig("logprob_distribution.png", dpi=300, bbox_inches='tight')

# # Histogram analysis
# print("Generating histogram of log probability values...")
# fig2, ax2 = plot_logprob_histogram(logprobs[0])
# fig2.savefig("logprob_histogram.png", dpi=300, bbox_inches='tight')

# # Top-k analysis
# print("Generating bar chart of top-10 token probabilities...")
# fig3, ax3 = plot_top_k_probs(logprobs[0], k=10)
# fig3.savefig("top_k_probs.png", dpi=300, bbox_inches='tight')


