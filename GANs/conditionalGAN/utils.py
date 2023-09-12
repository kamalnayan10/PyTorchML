import torch
from torch import nn

def gradient_penalty(critic , labels , real , fake , device = "cpu"):
    BATCH_SIZE , C , H , W = real.shape
    epsilon = torch.rand((BATCH_SIZE , 1 , 1 , 1)).repeat(1,C,H,W).to(device)

    interpolated_imgs = real*epsilon + fake*(1-epsilon)

    mixed_score = critic(interpolated_imgs , labels)

    gradient = torch.autograd.grad(
        inputs = interpolated_imgs,
        outputs= mixed_score , 
        grad_outputs=torch.ones_like(mixed_score),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient = gradient.view(gradient.shape[0],-1)
    gradient_norm = gradient.norm(2 , dim = 1)

    gradient_penalty = torch.mean((gradient_norm-1)**2)

    return gradient_penalty