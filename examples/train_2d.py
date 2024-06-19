import torch
from torch.distributions import MultivariateNormal
from difs import Unet, GaussianDiffusionConditional, DiFS

def evaluate(x):
    """
    Evaluation function for the 2D example.

    This is a simple example of the interface for the evaluation function.

    Args:
        x (torch.Tensor): The input tensor of shape (batch_size, xdim, horizon).

    Returns:
        torch.Tensor: The robustness associated with each sample, shape (batch_size,).
        torch.Tensor: The state trajectories associated with each disturbance, shape (batch_size, sdim, horizon). For logging only.
                      

    """
    s = x
    r = torch.maximum(3.0 - torch.min(torch.abs(s[:, :, 0]), s[:, :, 1]), torch.tensor(0.0)).squeeze()
    return r, s

# System Parameters
horizon = 2
xdim = 1

# Model Parameters
Nsamples = 10000
unet_dim = 32

# Initial disturbances
px = MultivariateNormal(torch.zeros(xdim), torch.eye(xdim))
initial_data = px.sample((Nsamples, horizon)).swapaxes(1, 2) # (Nsamples, xdim, horizon)


model = Unet(
    dim=unet_dim,
    dim_mults=(1, 2),
    channels=xdim,
    cond_dim=horizon,
)

diffusion = GaussianDiffusionConditional(
    model,
    seq_length=horizon,
)

trainer = DiFS(
    diffusion,
    evaluate_fn=evaluate,
    init_disturbances=initial_data,
    N=Nsamples,
    train_num_steps=10000,
    train_lr=1e-3,
)

# Train the model
trainer.train()

# Draw 100 samples from the model
conditions = torch.zeros(100)
samples = trainer.sample(conditions)

