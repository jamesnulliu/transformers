import torch

def weighted_forward_tp(
    topk_probs: torch.Tensor,
    topk_running_sequences: torch.Tensor,
    topk_running_beam_indices: torch.Tensor,
    embedding_table: torch.Tensor,
    ):
    """
    Tensor Parallel weighted embedding forward.

    Args:
        topk_probs: [B, K] tensor of probabilities for top-K tokens.
        topk_running_sequences: [B, K, S] tensor of running sequences for top-K tokens.
        topk_running_beam_indices: [B, K] tensor of beam indices for top-K tokens.
        embedding_table: [V, D] tensor of token embeddings.
    """
    # Step 1: Normalize probs to sum to 1
    topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True) # [B, K]
    # Step 2: Select embedding table rows based on topk_running_sequences
    
