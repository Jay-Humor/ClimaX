import torch

def generate_mask(shape, mask_prob=0.1):
    """Generate a random mask tensor.

    Args:
        shape (tuple): shape of the mask tensor.
        mask_prob (float): probability of an element to be masked.

    Returns:
        mask (torch.BoolTensor): the mask tensor, where True indicates the element should be masked.
    """
    mask = torch.rand(shape) < mask_prob
    return mask

def apply_mask(x, mask):
    """Apply mask to a tensor.

    Args:
        x (torch.Tensor): input tensor.
        mask (torch.BoolTensor): mask tensor. Must have the same shape as `x`.

    Returns:
        x_masked (torch.Tensor): masked tensor.
    """
    x_masked = x.clone()
    x_masked[mask] = 0
    return x_masked