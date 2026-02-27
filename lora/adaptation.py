import torch


def adapt_loss_fn(Ax, y):
    residual = Ax - y
    if Ax.dtype == torch.complex64:
        residual = torch.view_as_real(residual)
    loss = torch.mean(residual.pow(2))
    return loss

def adapt_loss_kl(Ax, y):
    Ax = torch.clamp(Ax, min=0) + 1e-7
    return torch.sum(Ax - y*torch.log(Ax))