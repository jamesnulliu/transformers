import torch
import matplotlib.pyplot as plt
import numpy as np


def setup_matplotlib_style():
    try:
        # Set global font to Times New Roman
        plt.rcParams.update(
            {
                "font.family": "serif",
                # "font.serif": ["Times New Roman"],
                "font.size": 12,
                "axes.labelsize": 14,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "legend.fontsize": 12,
                "figure.titlesize": 16,
            }
        )
    except Exception as e:
        print(f"Warning: Could not set font to 'Times New Roman'. Using default. Error: {e}")

    # Set other styling for a clean, academic look
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.7,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def visualize_logprob_line(logprob: torch.Tensor, title: str = "Log Probability Distribution"):
    """
    Visualizes a logprob tensor of shape (vocab_size,) as a line chart.

    Args:
        logprob (torch.Tensor): A 1D tensor of log probabilities.
        title (str): The title for the plot.

    Returns:
        tuple: A tuple containing the matplotlib figure and axes objects.
    """
    # Ensure tensor is on CPU and converted to numpy for plotting
    if logprob.is_cuda:
        logprob = logprob.cpu()
    logprob_np = logprob.detach().numpy()

    vocab_size = len(logprob_np)
    indices = np.arange(vocab_size)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Use a professional, muted color
    ax.plot(indices, logprob_np, color="#00529B", linewidth=1.5)

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Vocabulary Index")
    ax.set_ylabel("Log Probability")
    ax.set_xlim(0, vocab_size - 1)

    # Add a horizontal line at log(1/vocab_size) for reference (uniform distribution)
    uniform_logprob = -np.log(vocab_size)
    ax.axhline(
        y=uniform_logprob, color="gray", linestyle=":", linewidth=1.5, label=f"Uniform Dist. ({uniform_logprob:.2f})"
    )
    ax.legend()

    fig.tight_layout()
    return fig, ax


def plot_logprob_histogram(logprob: torch.Tensor, bins: int = 50, title: str = "Histogram of Log Probabilities"):
    """
    Creates a histogram to analyze the distribution of log probability values.

    Args:
        logprob (torch.Tensor): A 1D tensor of log probabilities.
        bins (int): The number of bins for the histogram.
        title (str): The title for the plot.

    Returns:
        tuple: A tuple containing the matplotlib figure and axes objects.
    """
    if logprob.is_cuda:
        logprob = logprob.cpu()
    logprob_np = logprob.detach().numpy()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Use a different color from the palette
    ax.hist(logprob_np, bins=bins, color="#6F8B93", edgecolor="black")

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Log Probability Value")
    ax.set_ylabel("Frequency (Count)")

    fig.tight_layout()
    return fig, ax


def plot_top_k_probs(logprob: torch.Tensor, k: int = 10, title: str = "Top-K Token Probabilities"):
    """
    Converts logprobs to probabilities and visualizes the top k most likely tokens.

    Args:
        logprob (torch.Tensor): A 1D tensor of log probabilities.
        k (int): The number of top tokens to display.
        title (str): The title for the plot.

    Returns:
        tuple: A tuple containing the matplotlib figure and axes objects.
    """
    # Convert log probabilities to probabilities
    probs = torch.exp(logprob)

    # Get the top k probabilities and their indices
    top_k_probs, top_k_indices = torch.topk(probs, k)

    # Move to CPU and numpy for plotting
    if top_k_probs.is_cuda:
        top_k_probs = top_k_probs.cpu()
        top_k_indices = top_k_indices.cpu()

    top_k_probs_np = top_k_probs.detach().numpy()
    top_k_indices_np = top_k_indices.detach().numpy()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bar chart
    y_pos = np.arange(len(top_k_indices_np))
    ax.barh(y_pos, top_k_probs_np, align="center", color="#D55E00", edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Index {i}" for i in top_k_indices_np])
    ax.invert_yaxis()  # Display the highest probability at the top

    ax.set_title(f"{title} (k={k})", fontweight="bold")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Token")

    # Add probability values on the bars
    for index, value in enumerate(top_k_probs_np):
        ax.text(value, index, f" {value:.3f}", va="center")

    ax.set_xlim(0, top_k_probs_np[0] * 1.15)  # Adjust x-limit for text

    fig.tight_layout()
    return fig, ax

