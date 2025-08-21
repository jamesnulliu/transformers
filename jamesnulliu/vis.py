from transformers.hack.vis import *  # noqa: F403

import torch
from matplotlib import pyplot as plt


if __name__ == "__main__":
    # --- Example Usage ---

    # 1. Set the global style
    setup_matplotlib_style()

    # 2. Create a sample log probability tensor
    # This simulates the output of a language model's final layer.
    vocab_size = 1000
    # Use randn for some variation and add a peak to make it interesting
    logits = torch.randn(vocab_size)
    logits[50] += 5  # Create a peak at index 50
    logits[250] += 3  # Another smaller peak

    # Convert logits to log probabilities using log_softmax
    log_probabilities = torch.nn.functional.log_softmax(logits, dim=0)

    # 3. Generate and display the plots

    # Main line chart visualization
    print("Generating line chart of all log probabilities...")
    fig1, ax1 = visualize_logprob_line(
        log_probabilities, title=f"Log Probability Distribution (Vocab Size: {vocab_size})"
    )
    # Save
    fig1.savefig("logprob_distribution.png", dpi=300, bbox_inches='tight')

    # Histogram analysis
    print("Generating histogram of log probability values...")
    fig2, ax2 = plot_logprob_histogram(log_probabilities)
    fig2.savefig("logprob_histogram.png", dpi=300, bbox_inches='tight')

    # Top-k analysis
    print("Generating bar chart of top-10 token probabilities...")
    fig3, ax3 = plot_top_k_probs(log_probabilities, k=10)
    fig3.savefig("top_k_probs.png", dpi=300, bbox_inches='tight')
