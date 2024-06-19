from typing import Callable, Optional
from pathlib import Path
from multiprocessing import cpu_count

import torch
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

from accelerate import Accelerator, DataLoaderConfiguration
from ema_pytorch import EMA

from difs.diffusion import GaussianDiffusionConditional
from difs.dataset import DatasetConditional
from difs.utils import exists, cycle

import wandb
from tqdm.auto import tqdm


class DiFS(object):
    def __init__(
        self,
        diffusion_model: GaussianDiffusionConditional,
        evaluate_fn: Callable, # should take in a tensor of shape (B, C, T) where B is batch-size, C is number of channels, and T is lenght of time series. Return a tuple of two tensors: the first tensor should be the robustness value, and the second tensor should be the state trajectory.
        init_disturbances: torch.Tensor,
        *,
        alpha: float = 0.5,
        N: int = 1000,
        max_iters: int = 10,
        train_num_steps: int = 10000,
        train_batch_size: int = 16,
        sample_batch_size: int = 10000,
        gradient_accumulate_every: int = 1,
        train_lr: float = 1e-3,
        ema_update_every: float = 10,
        ema_decay: float = 0.995,
        amp: bool = False,
        mixed_precision_type: str = 'fp16',
        split_batches: bool = True,
        max_grad_norm: float = 1.,
        experiment_name: str = "test",
        results_folder: str = './results',
        sample_callback: Optional[Callable] = None,
        use_wandb: bool = False,
        wandb_plot_fn: Optional[Callable] = None
    ):
        super().__init__()

        self.init_disturbances = init_disturbances
        self.rho_target = 0.0
        self.alpha = alpha
        self.N = N
        self.evaluate_fn = evaluate_fn
        self.train_batch_size = train_batch_size
        self.cond_dim = diffusion_model.model.cond_dim
        self.sample_batch_size = sample_batch_size
        self.use_wandb = use_wandb
        self.wandb_plot_fn = wandb_plot_fn
        self.max_iters = max_iters

        # accelerator
        self.accelerator = Accelerator(
            dataloader_config = DataLoaderConfiguration(split_batches=split_batches),
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model
        self.model = diffusion_model
        self.channels = diffusion_model.channels

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm

        self.train_num_steps = train_num_steps

        # optimizer
        #self.opt = Adam(diffusion_model.parameters(), lr=train_lr)
        self.opt = AdamW(diffusion_model.parameters(), lr=train_lr, weight_decay=1e-5)

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)
        
        # for logging results in a folder periodically
        self.results_folder = Path(results_folder+"/"+experiment_name)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state
        self.step = 0

        # prepare model, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        self.sample_callback = sample_callback

    @property
    def device(self):
        return self.accelerator.device

    def save(self, k):
        if not self.accelerator.is_local_main_process:
            return

        save_data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(save_data, str(self.results_folder / f'model-{k}.pt'))

    def load(self, k):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{k}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def log_wandb(self, rho, data, conds, samples, robustness, elite_samples, elite_rs, observations):
        info = {
            "rho": rho,
            "dataset_size": data.shape[0],
            "mean_sampled_risk": robustness.mean().cpu(),
            "min_sampled_risk": robustness.min().cpu(),
            "max_sampled_risk": robustness.max().cpu(),
            "std_sampled_risk": robustness.std().cpu(),
            "mean_elite_risk": elite_rs.mean().cpu(),
            "min_elite_risk": elite_rs.min().cpu(),
            "max_elite_risk": elite_rs.max().cpu(),
            "std_elite_risk": elite_rs.std().cpu(),
        }

        if self.wandb_plot_fn is not None:
            fig = self.wandb_plot_fn(observations, robustness, self.rho_target)
            info["samples"] = wandb.Image(fig)
            
        wandb.log(info)

    @torch.no_grad()
    def sample(self, rho_sample):
        rho_sample = rho_sample.unsqueeze(1).repeat(1, self.model.cond_dim).to(self.device)
        if rho_sample.shape[0] > self.sample_batch_size:
            rho_sample_tuple = torch.split(rho_sample, self.sample_batch_size, dim=0)
            samples = [self.model.sample(rs) for rs in rho_sample_tuple]
            samples = torch.cat(samples, dim=0)
        else:
            samples = self.model.sample(rho_sample)
        return samples

    def training_loop(self, data, conds):
        accelerator = self.accelerator
        device = accelerator.device
        
        dataset = DatasetConditional(data.cpu(), conds.cpu())
        dl = DataLoader(dataset, batch_size = self.train_batch_size, pin_memory = True, num_workers = cpu_count())
        dl = self.accelerator.prepare(dl)
        dl_cycle = cycle(dl)

        print("Training on updated data with {} samples...".format(data.shape[0]))
        step=0
        with tqdm(initial = step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:
                while step < self.train_num_steps:

                    total_loss = 0.

                    for _ in range(self.gradient_accumulate_every):
                        databatch, condbatch = next(dl_cycle)
                        databatch.to(device)
                        condbatch.to(device)

                        with self.accelerator.autocast():
                            loss = self.model(databatch, condbatch)
                            loss = loss / self.gradient_accumulate_every
                            total_loss += loss.item()

                        self.accelerator.backward(loss)

                    pbar.set_description(f'loss: {total_loss:.4f}')

                    accelerator.wait_for_everyone()
                    accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.opt.step()
                    self.opt.zero_grad()

                    accelerator.wait_for_everyone()

                    step += 1
                    if accelerator.is_main_process:
                        self.ema.update()

                    pbar.update(1)

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        data = self.init_disturbances.clone().detach()
        conds = self.evaluate_fn(data)[0].unsqueeze(1).repeat(1, self.cond_dim)

        self.training_loop(data, conds)

        rho = torch.quantile(conds, 1-self.alpha).item()

        k = 0
        while rho > self.rho_target and k <= self.max_iters:

            # Sample from the model with conditions [0, rho]
            print("rho = {}".format(rho))
            print("sampling from model...")
            rho_sample = torch.distributions.Uniform(-0.1, rho).sample((self.N,))
            rho_sample = torch.clamp(rho_sample, min=0.0)

            # Repeat rho_sample to be (1, self.cond_dim)
            # rho_sample = rho_sample.unsqueeze(1).repeat(1, self.model.cond_dim).to(device)

            # Draw samples
            samples = self.sample(rho_sample)

            # Evaluate robustness of each sample
            robustness, observations = self.evaluate_fn(samples)#[self.robustness(sample) for sample in samples]
            robustness = robustness.unsqueeze(1).repeat(1, self.cond_dim)

            # Compute the (1-alpha) quantile of robustness
            rho = torch.quantile(robustness, 1-self.alpha).item()
            rho = max(rho, self.rho_target)

            # Add samples and conditions to dataset
            data = torch.cat((data, samples.cpu()), dim=0)
            conds = torch.cat((conds, robustness.cpu()), dim=0)

            # Logging to wandb
            if self.use_wandb and wandb.run is not None:
                self.log_wandb(rho, data, conds, samples, robustness, samples, robustness, observations)

            # train
            mask = conds[:, 0] <= rho
            self.training_loop(data[mask, :, :], conds[mask, :])

            # Save off samples and robustness
            torch.save(samples, str(self.results_folder / f'samples-{k}.pt'))
            torch.save(robustness, str(self.results_folder / f'robustness-{k}.pt'))
            torch.save(observations, str(self.results_folder / f'observations-{k}.pt'))

            self.save(k)
            k += 1

        # Sample from the model with condition = 0
        rho_sample = torch.zeros(self.N).to(device)
        samples = self.sample(rho_sample)

        robustness, observations = self.evaluate_fn(samples)

        # Save off samples and robustness
        torch.save(samples, str(self.results_folder / f'samples-{k}.pt'))
        torch.save(robustness, str(self.results_folder / f'robustness-{k}.pt'))
        torch.save(observations, str(self.results_folder / f'observations-{k}.pt'))

        # save off final model
        self.save(k)