from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors. The rotation to each token
    embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    batch_size, seqlen, n_local_heads, _ = query.shape
    _, _, n_local_kv_heads, _ = key.shape

    print(n_local_heads, n_local_kv_heads)
    device = query.device
    # todo
    #
    # Please refer to slide 22 in https://phontron.com/class/anlp2024/assets/slides/anlp-05-transformers.pdf
    # and Section 3 in https://arxiv.org/abs/2104.09864.

    # reshape xq and xk to match the complex representation
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)
    # This separates each query/key vector into its odd and even indices (assuming *one-indexing*).
    # query_real contains q_1, q_3, q_5, ... and query_imag contains q_2, q_4, q_6, ...

    # First, compute the trigonometric values in the second and fourth columns in
    # slide 22 (linked above).

    # Calculate half of the dimension and frequency
    dim_half = head_dim // 2
    freq = 1.0 / (theta ** (torch.arange(0, dim_half, 2).float() / dim_half)).to(device)  # Shape: (dim_half//2,)


    positions = torch.arange(seqlen).float().unsqueeze(1)  # Shape: (seqlen, 1)
    angles = positions * freq.unsqueeze(0)  # Shape: (seqlen, dim_half//2)

    # Correct shape for broadcasting
    cos_vals = torch.cos(angles).repeat(1, 2)  # Shape: (seqlen, dim_half)
    sin_vals = torch.sin(angles).repeat(1, 2)  # Shape: (seqlen, dim_half)

    cos_vals = torch.cos(angles).unsqueeze(0).unsqueeze(2).expand(batch_size, seqlen, 2, dim_half)
    sin_vals = torch.sin(angles).unsqueeze(0).unsqueeze(2).expand(batch_size, seqlen, 2, dim_half)



    # Reshape query and key to separate real/imaginary parts
    query_real, query_imag = query.reshape(batch_size, seqlen, n_local_heads, -1, 2).unbind(-1)
    key_real, key_imag = key.reshape(batch_size, seqlen, n_local_kv_heads, -1, 2).unbind(-1)


    # Apply rotary embeddings
    query_out_real = query_real * cos_vals - query_imag * sin_vals
    query_out_imag = query_real * sin_vals + query_imag * cos_vals

    key_out_real = key_real * cos_vals - key_imag * sin_vals
    key_out_imag = key_real * sin_vals + key_imag * cos_vals

    # Fix duplicated rows: Properly reshape the stacked tensor
    query_out = torch.stack((query_out_real, query_out_imag), dim=-1).reshape(batch_size, seqlen, n_local_heads, head_dim)
    key_out = torch.stack((key_out_real, key_out_imag), dim=-1).reshape(batch_size, seqlen, n_local_kv_heads, head_dim)

    print(query_out[:, :, 0, :])  # First head
    print(query_out[:, :, 1, :])  # Second head






    # Return the rotary position embeddings for the query and key tensors
    return query_out, key_out