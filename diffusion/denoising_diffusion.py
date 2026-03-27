# -*- coding: UTF-8 -*-
import math
import copy
from pathlib import Path
import random 
from functools import partial
from collections import namedtuple, Counter
from multiprocessing import cpu_count
import os
import numpy as np
import csv
import timeit
import argparse
import json
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import AdamW

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from tqdm.auto import tqdm
from ema_pytorch import EMA

from transformers import get_scheduler,T5ForConditionalGeneration,T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput
from accelerate import Accelerator
import wandb

import diffusion.constant as constant
import diffusion.optimizer as optimizer
import dataset_utils.dataset as dataset
from utils.torch_utils import compute_grad_norm
import utils.file_utils as file_utils
from evaluation import evaluation
import enum
from latent_models.latent_utils import get_latent_model


ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start', 'pred_v'])
# helpers functions
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def l2norm(t):
    return F.normalize(t, dim = -1)

def log(t, eps = 1e-12):
    return torch.log(t.clamp(min = eps))

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# normalize variance of noised latent, if scale is not 1

def normalize_z_t_variance(z_t, mask, eps = 1e-5):
    std = rearrange([reduce(z_t[i][:torch.sum(mask[i])], 'l d -> 1 1', partial(torch.std, unbiased = False)) for i in range(z_t.shape[0])], 'b 1 1 -> b 1 1')
    return z_t / std.clamp(min = eps)
    

# noise schedules

def simple_linear_schedule(t, clip_min = 1e-9):
    return (1 - t).clamp(min = clip_min)

def beta_linear_schedule(t, clip_min = 1e-9):
    return torch.exp(-1e-4 - 10 * (t ** 2)).clamp(min = clip_min, max = 1.)

def cosine_schedule(t, start = 0, end = 1, tau = 1, clip_min = 1e-9):
    power = 2 * tau
    v_start = math.cos(start * math.pi / 2) ** power
    v_end = math.cos(end * math.pi / 2) ** power
    output = torch.cos((t * (end - start) + start) * math.pi / 2) ** power
    output = (v_end - output) / (v_end - v_start)
    return output.clamp(min = clip_min)

def sigmoid_schedule(t, start = -3, end = 3, tau = 1, clamp_min = 1e-9):
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    gamma = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    return gamma.clamp_(min = clamp_min, max = 1.)

# converting gamma to alpha, sigma or logsnr

def log_snr_to_alpha(log_snr):
    alpha = torch.sigmoid(log_snr)
    return alpha

def alpha_to_shifted_log_snr(alpha, scale = 1):
    return log((alpha / (1 - alpha))).clamp(min=-15, max=15) + 2*np.log(scale).item()

def time_to_alpha(t, alpha_schedule, scale):
    alpha = alpha_schedule(t)
    shifted_log_snr = alpha_to_shifted_log_snr(alpha, scale = scale)
    return log_snr_to_alpha(shifted_log_snr)

def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        max_seq_len,
        sampling_timesteps = 250,
        loss_type = 'l1',
        objective = 'pred_noise',
        train_schedule = 'cosine',
        sampling_schedule = None,
        scale = 1.,
        sampler = 'ddpm',
        train_prob_self_cond = 0.5,
    ):
        super().__init__()
        assert sampler in {'ddim', 'ddpm', 'dpmpp'}, 'sampler must be one of ddim, ddpm, dpmpp'
        self.sampler = sampler

        self.diffusion_model = model
        if self.diffusion_model.class_conditional:
            if self.diffusion_model.class_unconditional_prob > 0:
                self.class_unconditional_bernoulli = torch.distributions.Bernoulli(probs=self.diffusion_model.class_unconditional_prob)

        self.latent_dim = self.diffusion_model.latent_dim
        self.self_condition = self.diffusion_model.self_condition

        self.max_seq_len = max_seq_len
        self.l2_normalize = False

        self.objective = objective

        self.loss_type = loss_type

        assert objective in {'pred_noise', 'pred_x0', 'pred_v', 'pred_v_dual'}, 'objective must be one of pred_noise, pred_x0, pred_v, pred_v_dual'

        if train_schedule == "simple_linear":
            alpha_schedule = simple_linear_schedule
        elif train_schedule == "beta_linear":
            alpha_schedule = beta_linear_schedule
        elif train_schedule == "cosine":
            alpha_schedule = cosine_schedule
        elif train_schedule == "sigmoid":
            alpha_schedule = sigmoid_schedule
        else:
            raise ValueError(f'invalid noise schedule {train_schedule}')
        
        self.train_schedule = partial(time_to_alpha, alpha_schedule=alpha_schedule, scale=scale)

        # Sampling schedule
        if sampling_schedule is None:
            sampling_alpha_schedule = None
        elif sampling_schedule == "simple_linear":
            sampling_alpha_schedule = simple_linear_schedule
        elif sampling_schedule == "beta_linear":
            sampling_alpha_schedule = beta_linear_schedule
        elif sampling_schedule == "cosine":
            sampling_alpha_schedule = cosine_schedule
        elif sampling_schedule == "sigmoid":
            sampling_alpha_schedule = sigmoid_schedule
        else:
            raise ValueError(f'invalid sampling schedule {sampling_schedule}')
        
        if exists(sampling_alpha_schedule):
            self.sampling_schedule = partial(time_to_alpha, alpha_schedule=sampling_alpha_schedule, scale=scale)
        else:
            self.sampling_schedule = self.train_schedule

        # the main finding presented in Ting Chen's paper - that higher resolution images requires more noise for better training

        
        self.scale = scale

        # gamma schedules

        self.sampling_timesteps = sampling_timesteps

        # probability for self conditioning during training

        self.train_prob_self_cond = train_prob_self_cond

        # Buffers for latent mean and scale values
        self.register_buffer('latent_mean', torch.tensor([0]*self.latent_dim).to(torch.float32))
        self.register_buffer('latent_scale', torch.tensor(1).to(torch.float32))

    def predict_start_from_noise(self, z_t, t, noise, sampling=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        alpha = time_to_alpha(t)
        alpha = right_pad_dims_to(z_t, alpha)

        return (z_t - (1-alpha).sqrt() * noise) / alpha.sqrt().clamp(min = 1e-8)
        
    def predict_noise_from_start(self, z_t, t, x0, sampling=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        alpha = time_to_alpha(t)
        alpha = right_pad_dims_to(z_t, alpha)

        return (z_t - alpha.sqrt() * x0) / (1-alpha).sqrt().clamp(min = 1e-8)

    def predict_start_from_v(self, z_t, t, v, sampling=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        alpha = time_to_alpha(t)
        alpha = right_pad_dims_to(z_t, alpha)

        x = alpha.sqrt() * z_t - (1-alpha).sqrt() * v

        return x
    
    def predict_noise_from_v(self, z_t, t, v, sampling=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        alpha = time_to_alpha(t)
        alpha = right_pad_dims_to(z_t, alpha)

        eps = (1-alpha).sqrt() * z_t + alpha.sqrt() * v

        return eps
    
    def predict_v_from_start_and_eps(self, z_t, t, x, noise, sampling=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        alpha = time_to_alpha(t)
        alpha = right_pad_dims_to(z_t, alpha)

        v = alpha.sqrt() * noise - x* (1-alpha).sqrt()

        return v

    def normalize_latent(self, x_start):
        eps = 1e-5 
                
        return (x_start-self.latent_mean)/(self.latent_scale).clamp(min=eps)
    
    def unnormalize_latent(self, x_start):
        eps = 1e-5 
        
        return x_start*(self.latent_scale.clamp(min=eps))+self.latent_mean

    def diffusion_model_predictions(self, z_t, mask, t, *, x_self_cond = None,  class_id=None, sampling=False, cls_free_guidance=1.0, l2_normalize=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        time_cond = time_to_alpha(t)
        model_output = self.diffusion_model(z_t, mask, time_cond, x_self_cond, class_id=class_id)
        if cls_free_guidance!=1.0:
            if exists(class_id):
                unc_class_id = torch.full_like(class_id, fill_value=self.diffusion_model.num_classes)
            else:
                unc_class_id = None
            unc_model_output = self.diffusion_model(z_t, mask, time_cond, x_self_cond, class_id=unc_class_id)
            model_output = model_output*cls_free_guidance + unc_model_output*(1-cls_free_guidance)

        pred_v = None
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(z_t, t, pred_noise, sampling=sampling)
        elif self.objective =='pred_x0':
            x_start = model_output
            pred_noise = self.predict_noise_from_start(z_t, t, x_start, sampling=sampling)
            pred_v = self.predict_v_from_start_and_eps(z_t, t, x_start, pred_noise, sampling=sampling)
        elif self.objective == 'pred_v':
            pred_v = model_output
            x_start = self.predict_start_from_v(z_t, t, pred_v, sampling=sampling)
            pred_noise = self.predict_noise_from_v(z_t, t, pred_v, sampling=sampling)
        else:
            raise ValueError(f'invalid objective {self.objective}')
        if l2_normalize:
            assert sampling
            x_start = F.normalize(x_start, dim=-1) * math.sqrt(x_start.shape[-1])
            pred_noise = self.predict_noise_from_start(z_t, t, x_start, sampling=sampling)
            pred_v = self.predict_v_from_start_and_eps(z_t, t, x_start, pred_noise, sampling=sampling)

        return ModelPrediction(pred_noise, x_start, pred_v)

    def get_sampling_timesteps(self, batch, *, device, invert = False):
        times = torch.linspace(1., 0., self.sampling_timesteps + 1, device = device)
        if invert:
            times = times.flip(dims = (0,))
        times = repeat(times, 't -> b t', b = batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim = 0)
        times = times.unbind(dim = -1)
        return times

    @torch.no_grad()
    def ddim_sample(self, shape, lengths, class_id,cls_free_guidance=1.0, l2_normalize=False, invert=False, z_t=None):
        print('DDIM sampling')
        batch, device = shape[0], next(self.diffusion_model.parameters()).device

        time_pairs = self.get_sampling_timesteps(batch, device = device, invert=invert)
        if invert:
            assert exists(z_t)

        if not exists(z_t):
            z_t = torch.randn(shape, device=device)

        x_start = None
        latent=None
        if self.using_latent_model:
            mask = torch.ones((shape[0], shape[1]), dtype=torch.bool, device=device)
        else:    
            mask = [[True]*length + [False]*(self.max_seq_len-length) for length in lengths]
            mask = torch.tensor(mask, dtype=torch.bool, device=device)

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step', total = self.sampling_timesteps):
            # get predicted x0

            model_output = self.diffusion_model_predictions(z_t, mask, time, class_id=class_id, x_self_cond=x_start, sampling=True, cls_free_guidance=cls_free_guidance, l2_normalize=l2_normalize)
            # get alpha sigma of time and next time

            alpha = self.sampling_schedule(time)
            alpha_next = self.sampling_schedule(time_next)
            alpha, alpha_next = map(partial(right_pad_dims_to, z_t), (alpha, alpha_next))

            # # calculate x0 and noise

            x_start = model_output.pred_x_start

            eps = model_output.pred_noise

            
            if (not invert) and time_next[0] <= 0:
                z_t = x_start
                continue
            if invert and time_next[0] >= 1:
                z_t = eps
                continue
            
            # get noise
            
            z_t = x_start * alpha_next.sqrt() + eps * (1-alpha_next).sqrt()
        return (z_t, mask)


    @torch.no_grad()
    def ddpm_sample(self, shape, lengths, class_id, cls_free_guidance=1.0, l2_normalize=False, invert=False, z_t=None):
        batch, device = shape[0], next(self.diffusion_model.parameters()).device

        time_pairs = self.get_sampling_timesteps(batch, device = device)

        if not exists(z_t):
            z_t = torch.randn(shape, device=device)

        x_start = None
        latent=None
        if self.using_latent_model:
            mask = torch.ones((shape[0], shape[1]), dtype=torch.bool, device=device)
        else:    
            mask = [[True]*length + [False]*(self.max_seq_len-length) for length in lengths]
            mask = torch.tensor(mask, dtype=torch.bool, device=device)

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step', total = self.sampling_timesteps):
            # get predicted x0

            model_output = self.diffusion_model_predictions(z_t, mask, time, class_id=class_id, x_self_cond=x_start,sampling=True, cls_free_guidance=cls_free_guidance, l2_normalize=l2_normalize)
            # get alpha sigma of time and next time

            alpha = self.sampling_schedule(time)
            alpha_next = self.sampling_schedule(time_next)
            alpha, alpha_next = map(partial(right_pad_dims_to, z_t), (alpha, alpha_next))

            alpha_now = alpha/alpha_next

            # # calculate x0 and noise

            x_start = model_output.pred_x_start

            eps = model_output.pred_noise
            
            if time_next[0] <= 0:
                z_t = x_start
                continue         
            
            # get noise

            noise = torch.randn_like(z_t)
            
            z_t = 1/alpha_now.sqrt() * (z_t - (1-alpha_now)/(1-alpha).sqrt() * eps) + torch.sqrt(1 - alpha_now) * noise
        return (z_t, mask)
    

    @torch.no_grad()
    def dpmpp_sample(self, shape, lengths, class_id, cls_free_guidance=1.0, l2_normalize=False, invert=False, z_t=None):
        batch, device = shape[0], next(self.diffusion_model.parameters()).device

        time_pairs = self.get_sampling_timesteps(batch, device = device)

        if not exists(z_t):
            z_t = torch.randn(shape, device=device)

        x_start = None
        latent=None
        if self.using_latent_model:
            mask = torch.ones((shape[0], shape[1]), dtype=torch.bool, device=device)
        else:    
            mask = [[True]*length + [False]*(self.max_seq_len-length) for length in lengths]
            mask = torch.tensor(mask, dtype=torch.bool, device=device)

        old_pred_x = []
        old_hs = []

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step', total = self.sampling_timesteps):
            # get predicted x0

            model_output = self.diffusion_model_predictions(z_t, mask, time, class_id=class_id, x_self_cond=x_start, sampling=True, cls_free_guidance=cls_free_guidance, l2_normalize=l2_normalize)
            # get alpha sigma of time and next time

            alpha = self.sampling_schedule(time)
            alpha_next = self.sampling_schedule(time_next)
            alpha, alpha_next = map(partial(right_pad_dims_to, z_t), (alpha, alpha_next))
            sigma, sigma_next = 1-alpha, 1-alpha_next

            alpha_now = alpha/alpha_next

            lambda_now = ((log(alpha) - log(1-alpha))/2)
            lambda_next = ((log(alpha_next) - log(1-alpha_next))/2)
            h = lambda_next - lambda_now

            # calculate x0 and noise
            if time_next[0] <= 0:
                z_t = x_start
                continue  

            x_start = model_output.pred_x_start

            phi_1 = torch.expm1(-h)
            if len(old_pred_x) < 2:
                denoised_x = x_start
            else:
                h = lambda_next - lambda_now
                h_0 = old_hs[-1]
                r0 = h_0/h
                gamma = -1/(2*r0)
                denoised_x = (1-gamma)*x_start + gamma*old_pred_x[-1]
            
            z_t = (sigma_next.sqrt()/sigma.sqrt()) * z_t - alpha_next.sqrt() * phi_1 * denoised_x
        return (z_t, mask)
    

    @torch.no_grad()
    def sample(self, batch_size, length, class_id=None, cls_free_guidance=1.0, l2_normalize=False):
        max_seq_len, latent_dim = self.max_seq_len, self.latent_dim
        
        if self.sampler == 'ddim':
            sample_fn = self.ddim_sample
        elif self.sampler == 'ddpm':
            sample_fn = self.ddpm_sample
        elif self.sampler == 'dpmpp':
            sample_fn = self.dpmpp_sample
        else:
            raise ValueError(f'invalid sampler {self.sampler}')
        return sample_fn((batch_size, max_seq_len, latent_dim), length, class_id, cls_free_guidance, l2_normalize)

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        elif self.loss_type == 'smooth_l1':
            return F.smooth_l1_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def forward(self, txt_latent, mask, class_id, return_x_start=False, *args, **kwargs):
        batch, l, d, device, max_seq_len, = *txt_latent.shape, txt_latent.device, self.max_seq_len
        assert l == max_seq_len, f'length must be {self.max_seq_len}'
        
        # sample random times

        times = torch.zeros((batch,), device = device).float().uniform_(0, 1.)
        # noise sample

        noise = torch.randn_like(txt_latent)

        alpha = self.train_schedule(times)
        alpha = right_pad_dims_to(txt_latent, alpha)

        z_t = alpha.sqrt() * txt_latent + (1-alpha).sqrt() * noise

        # Perform unconditional generation with some probability
        if self.diffusion_model.class_conditional and self.diffusion_model.class_unconditional_prob > 0:
            assert exists(class_id)
            class_unconditional_mask = self.class_unconditional_bernoulli.sample(class_id.shape).bool()
            class_id[class_unconditional_mask] = self.diffusion_model.num_classes

        self_cond = None

        if self.self_condition and (random.random() < self.train_prob_self_cond):
            with torch.no_grad():
                model_output = self.diffusion_model_predictions(z_t, mask, times, class_id=class_id)
                self_cond = model_output.pred_x_start.detach()
                if self.l2_normalize:
                    self_cond = F.normalize(self_cond, dim=-1) * math.sqrt(self_cond.shape[-1])
              

        # predict and take gradient step

        predictions = self.diffusion_model_predictions(z_t, mask, times, x_self_cond=self_cond, class_id=class_id)          
        if self.objective == 'pred_x0':
            target = txt_latent
            pred = predictions.pred_x_start
        elif self.objective == 'pred_noise':
            target = noise
            pred = predictions.pred_noise
        elif self.objective == 'pred_v':
            target = alpha.sqrt() * noise - (1-alpha).sqrt() * txt_latent
            assert exists(predictions.pred_v)
            pred = predictions.pred_v
            
        loss = self.loss_fn(pred, target, reduction = 'none')
        loss = rearrange([reduce(loss[i][:torch.sum(mask[i])], 'l d -> 1', 'mean') for i in range(txt_latent.shape[0])], 'b 1 -> b 1')


        if return_x_start:
            return loss.mean(), predictions.pred_x_start
        return loss.mean()


# trainer class

class Trainer(object):
    def __init__(
        self,
        args,
        diffusion,
        dataset_name,
        *,
        train_batch_size = 16,
        eval_batch_size = 64,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        lr_schedule = 'cosine',
        num_warmup_steps = 500,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        adam_weight_decay = 0.01,
        save_and_sample_every = 5000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision = 'no',
        split_batches = True,
    ):
        super().__init__()

        set_seeds(42)
        self.args = args
        self.train_lr = train_lr

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision
        )

        if self.accelerator.is_main_process:
            run = os.path.split(__file__)[-1].split(".")[0]
            if args.wandb_name:
                self.accelerator.init_trackers(run, config=args, init_kwargs={"wandb": {"dir": results_folder, "name": args.wandb_name}})
            else:
                self.accelerator.init_trackers(run, config=args, init_kwargs={"wandb": {"dir": results_folder}})

        self.accelerator.native_amp = amp

        self.diffusion = diffusion

        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps

        self.latent_model_path = args.latent_model_path

        self.max_seq_len = diffusion.max_seq_len
        
        self.class_conditional = self.diffusion.diffusion_model.class_conditional

        # Init T5 model 
        self.t5_model = T5ForConditionalGeneration.from_pretrained(args.enc_dec_model, torch_dtype=torch.bfloat16)
        self.tokenizer = T5Tokenizer.from_pretrained('PLM/prot_t5_xl_uniref50', do_lower_case=False)

        self.diffusion.using_latent_model = False
        self.context_tokenizer = None

        if args.latent_model_path:
            device = self.accelerator.device
            with open(os.path.join(args.latent_model_path, 'args.json'), 'rt') as f:
                latent_model_args = json.load(f)
            
            latent_argparse = argparse.Namespace(**latent_model_args)
            self.diffusion.context_encoder = self.t5_model.get_encoder()
            #self.seq2seq_train_context_encoder = seq2seq_train_context_encoder
            #if seq2seq_train_context_encoder:
            #    for param in self.diffusion.context_encoder.parameters():
            #        param.requires_grad = True
            #else:
                # for param in self.diffusion.context_encoder.parameters():
                #     param.requires_grad = False

            for param in self.diffusion.context_encoder.parameters():
                param.requires_grad = False

            self.context_tokenizer = self.tokenizer
            self.t5_model, self.tokenizer, _ = get_latent_model(latent_argparse)
            data = torch.load(os.path.join(args.latent_model_path, 'model.pt'), map_location=device)
            self.t5_model.load_state_dict(data['model'])
            self.diffusion.max_seq_len = self.t5_model.num_encoder_latents
            self.num_encoder_latents = self.t5_model.num_encoder_latents
            self.diffusion.using_latent_model = True
            self.diffusion.l2_normalize = (hasattr(self.t5_model, 'l2_normalize_latents') and self.t5_model.l2_normalize_latents)
            if self.diffusion.l2_normalize:
                assert not args.normalize_latent
            for param in self.t5_model.parameters():
                param.requires_grad = False
        self.using_latent_model = self.diffusion.using_latent_model
        self.t5_model.eval()

        # dataset and dataloader
        dataset = dataset.get_dataset(
            dataset_name,
            path=args.data_path,
        )

        self.dataset = dataset.shuffle(seed=42)
        self.num_samples = min(self.num_samples,len(self.dataset['valid']['AASeq']))
        # Subsample train and val splits for computing language generation during runtime
        self.dataloader = dataset.get_dataloader(args, dataset['train'], self.t5_model.config, self.tokenizer, self.max_seq_len)
        self.val_dataloader = dataset.get_dataloader(args, dataset['valid'], self.t5_model.config, self.tokenizer, self.max_seq_len)

        training_lengths = [min(sum(self.dataloader.dataset[idx]['attention_mask']), self.max_seq_len) for idx in range(self.dataloader.dataset.num_rows)]

        length_counts = Counter(training_lengths)

        probs = torch.tensor([length_counts[idx]/self.dataloader.dataset.num_rows for idx in range(self.max_seq_len+1)])

        assert probs[0] == 0, 'Can\'t have examples of length 0'
        self.length_categorical = torch.distributions.Categorical(probs=probs)

        if self.diffusion.diffusion_model.class_conditional:
            training_labels = [self.dataloader.dataset[idx]['Label'] for idx in range(self.dataloader.dataset.num_rows)]
            label_counts = Counter(training_labels)
            probs = torch.tensor([label_counts[idx]/self.dataloader.dataset.num_rows for idx in range(self.diffusion.diffusion_model.num_classes)])
            self.class_categorical = torch.distributions.Categorical(probs=probs)
        
        # optimizer

        self.opt = optimizer.get_adamw_optimizer(diffusion.parameters(), lr = train_lr, betas = adam_betas, weight_decay=adam_weight_decay)

        # scheduler

        lr_scheduler = get_scheduler(
            lr_schedule,
            optimizer=self.opt,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=train_num_steps,
        )

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion, beta = ema_decay, update_every = ema_update_every, power=3/4)

            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.diffusion, self.t5_model, self.opt, self.dataloader, self.lr_scheduler, self.val_dataloader = self.accelerator.prepare(self.diffusion, self.t5_model, self.opt, self.dataloader, lr_scheduler, self.val_dataloader)

        self.data_iter = cycle(self.dataloader)
        self.val_iter = cycle(self.val_dataloader)

    def save(self, best=False):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.diffusion),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'scheduler': self.lr_scheduler.state_dict(),
        }
        if best:
            torch.save(data, str(self.results_folder / f'best_model.pt'))
        else:
            torch.save(data, str(self.results_folder / f'model.pt'))


    def load(self, file_path=None, best=False, init_only=False):
        file_path = Path(file_path) if exists(file_path) else self.results_folder
        accelerator = self.accelerator
        device = accelerator.device

        if best:
            data = torch.load(str(file_path / f'best_model.pt'), map_location=device)
        else:
            data = torch.load(str(file_path / f'model.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.diffusion)
        # For backwards compatibility with earlier models
        model.load_state_dict(data['model'])
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_local_main_process:
            self.ema.load_state_dict(data['ema'])
        if init_only:
            return
        self.step = data['step']
        
        if 'scheduler' in data:
            self.lr_scheduler.load_state_dict(data['scheduler'])

        # resume training
        if self.args.resume_training:
            self.lr_scheduler._last_lr = [self.train_lr, self.train_lr]
        
        # For backwards compatibility with earlier models
        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def gen_cdr3(self,latents,device):
        batch_size = latents.size()[0]
        encoder_output = BaseModelOutput(last_hidden_state=self.t5_model.get_decoder_input(latents.clone()))
        max_length = 10
        input_ids = []
        input_ids.extend(self.tokenizer.encode("<pad>")[:-1])
        input_length = len(input_ids)

        input_tensor = torch.zeros(batch_size, input_length).long()
        input_tensor[:] = torch.tensor(input_ids)
        finished = torch.zeros(batch_size, input_length).byte().to(device)

        for i in range(max_length):
            logits = self.t5_model(encoder_outputs=encoder_output,decoder_input_ids=input_tensor.to(device)).logits
            
            # temp
            logits /= 1

            logits = F.softmax(logits[:, -1, :])
            last_token_id = torch.multinomial(logits, 1)

            EOS_sampled = (last_token_id == self.tokenizer.eos_token_id)
            finished = torch.ge(finished + EOS_sampled, 1)
            if torch.prod(finished) == 1:
                print('End')
                break

            input_tensor = torch.cat((input_tensor, last_token_id.detach().to('cpu')), 1)
        return self.tokenizer.batch_decode(input_tensor, skip_special_tokens=True)

    def gen_cdr3_12WG(self,latents,device):
        batch_size = latents.size()[0]

        encoder_output = BaseModelOutput(last_hidden_state=self.t5_model.get_decoder_input(latents.clone()))

        input_ids = []
        input_ids.extend(self.tokenizer.encode("<pad> W G")[:-1])
        input_length = len(input_ids)

        input_tensor = torch.zeros(batch_size, input_length).long()
        input_tensor[:] = torch.tensor(input_ids)
        finished = torch.zeros(batch_size, input_length).byte().to(device)

        for i in range(8):
            logits = self.t5_model(encoder_outputs=encoder_output,decoder_input_ids=input_tensor.to(device)).logits

            logits = F.softmax(logits[:, -1, :])
            last_token_id = torch.multinomial(logits, 1)

            EOS_sampled = (last_token_id == self.tokenizer.eos_token_id)
            finished = torch.ge(finished + EOS_sampled, 1)
            if torch.prod(finished) == 1:
                print('End')
                break
            input_tensor = torch.cat((input_tensor, last_token_id.detach().to('cpu')), 1)
            
        return self.tokenizer.batch_decode(input_tensor, skip_special_tokens=True)

    def gen_cdr3_12YG(self,latents,device):
        batch_size = latents.size()[0]
        encoder_output = BaseModelOutput(last_hidden_state=self.t5_model.get_decoder_input(latents.clone()))
        max_length = 10
        input_ids = []
        input_ids.extend(self.tokenizer.encode("<pad> Y G")[:-1])
        input_length = len(input_ids)

        input_tensor = torch.zeros(batch_size, input_length).long()
        input_tensor[:] = torch.tensor(input_ids)
        finished = torch.zeros(batch_size, input_length).byte().to(device)

        for i in range(8):
            logits = self.t5_model(encoder_outputs=encoder_output,decoder_input_ids=input_tensor.to(device)).logits

            logits = F.softmax(logits[:, -1, :])
            last_token_id = torch.multinomial(logits, 1)

            EOS_sampled = (last_token_id == self.tokenizer.eos_token_id)
            finished = torch.ge(finished + EOS_sampled, 1)
            if torch.prod(finished) == 1:
                print('End')
                break
            input_tensor = torch.cat((input_tensor, last_token_id.detach().to('cpu')), 1)
            
            
        return self.tokenizer.batch_decode(input_tensor, skip_special_tokens=True)
    
    def gen_cdr3_123WGG(self,latents,device):
        batch_size = latents.size()[0]
        encoder_output = BaseModelOutput(last_hidden_state=self.t5_model.get_decoder_input(latents.clone()))
        max_length = 10
        input_ids = []
        input_ids.extend(self.tokenizer.encode("<pad> W G G")[:-1])
        input_length = len(input_ids)

        input_tensor = torch.zeros(batch_size, input_length).long()
        input_tensor[:] = torch.tensor(input_ids)
        finished = torch.zeros(batch_size, input_length).byte().to(device)

        for i in range(7):
            logits = self.t5_model(encoder_outputs=encoder_output,decoder_input_ids=input_tensor.to(device)).logits

            logits = F.softmax(logits[:, -1, :])
            last_token_id = torch.multinomial(logits, 1)

            EOS_sampled = (last_token_id == self.tokenizer.eos_token_id)
            finished = torch.ge(finished + EOS_sampled, 1)
            if torch.prod(finished) == 1:
                print('End')
                break
            input_tensor = torch.cat((input_tensor, last_token_id.detach().to('cpu')), 1)
            
            
        return self.tokenizer.batch_decode(input_tensor, skip_special_tokens=True)
    
    def gen_cdr3_4567DGFY(self,latents,device):
        batch_size = latents.size()[0]
        encoder_output = BaseModelOutput(last_hidden_state=self.t5_model.get_decoder_input(latents.clone()))
        max_length = 10
        input_ids = []
        input_ids.extend(self.tokenizer.encode("<pad>")[:-1])
        input_length = len(input_ids)

        input_tensor = torch.zeros(batch_size, input_length).long()
        input_tensor[:] = torch.tensor(input_ids)
        finished = torch.zeros(batch_size, input_length).byte().to(device)

        for i in range(3):
            logits = self.t5_model(encoder_outputs=encoder_output,decoder_input_ids=input_tensor.to(device)).logits

            logits = F.softmax(logits[:, -1, :])
            last_token_id = torch.multinomial(logits, 1)

            EOS_sampled = (last_token_id == self.tokenizer.eos_token_id)
            finished = torch.ge(finished + EOS_sampled, 1)
            if torch.prod(finished) == 1:
                print('End')
                break
            input_tensor = torch.cat((input_tensor, last_token_id.detach().to('cpu')), 1)
            
        insert_ids = self.tokenizer.encode('D G F Y', add_special_tokens=False)
        insert_tensor = torch.tensor(insert_ids).repeat(batch_size).reshape(batch_size, -1)
        input_tensor = torch.cat((input_tensor, insert_tensor), 1)
        
        for i in range(3):
            logits = self.t5_model(encoder_outputs=encoder_output,decoder_input_ids=input_tensor.to(device)).logits

            logits = F.softmax(logits[:, -1, :])
            last_token_id = torch.multinomial(logits, 1)

            EOS_sampled = (last_token_id == self.tokenizer.eos_token_id)
            finished = torch.ge(finished + EOS_sampled, 1)
            if torch.prod(finished) == 1:
                print('End')
                break

            input_tensor = torch.cat((input_tensor, last_token_id.detach().to('cpu')), 1)
            
        return self.tokenizer.batch_decode(input_tensor, skip_special_tokens=True)
    
    def gen_cdr3_567GFY(self,latents,device):
        batch_size = latents.size()[0]
        encoder_output = BaseModelOutput(last_hidden_state=self.t5_model.get_decoder_input(latents.clone()))
        max_length = 10
        input_ids = []
        input_ids.extend(self.tokenizer.encode("<pad>")[:-1])
        input_length = len(input_ids)

        input_tensor = torch.zeros(batch_size, input_length).long()
        input_tensor[:] = torch.tensor(input_ids)
        finished = torch.zeros(batch_size, input_length).byte().to(device)

        for i in range(4):
            logits = self.t5_model(encoder_outputs=encoder_output,decoder_input_ids=input_tensor.to(device)).logits

            logits = F.softmax(logits[:, -1, :])
            last_token_id = torch.multinomial(logits, 1)

            EOS_sampled = (last_token_id == self.tokenizer.eos_token_id)
            finished = torch.ge(finished + EOS_sampled, 1)
            if torch.prod(finished) == 1:
                print('End')
                break
            input_tensor = torch.cat((input_tensor, last_token_id.detach().to('cpu')), 1)
            
        insert_ids = self.tokenizer.encode('G F Y', add_special_tokens=False)
        insert_tensor = torch.tensor(insert_ids).repeat(batch_size).reshape(batch_size, -1)
        input_tensor = torch.cat((input_tensor, insert_tensor), 1)
        
        for i in range(3):
            logits = self.t5_model(encoder_outputs=encoder_output,decoder_input_ids=input_tensor.to(device)).logits
            logits = F.softmax(logits[:, -1, :])
            last_token_id = torch.multinomial(logits, 1)

            EOS_sampled = (last_token_id == self.tokenizer.eos_token_id)
            finished = torch.ge(finished + EOS_sampled, 1)
            if torch.prod(finished) == 1:
                print('End')
                break

            input_tensor = torch.cat((input_tensor, last_token_id.detach().to('cpu')), 1)
            
        return self.tokenizer.batch_decode(input_tensor, skip_special_tokens=True)

    def gen_cdr3_56789GFYAF(self,latents,device):
        batch_size = latents.size()[0]
        encoder_output = BaseModelOutput(last_hidden_state=self.t5_model.get_decoder_input(latents.clone()))
        max_length = 10
        input_ids = []
        input_ids.extend(self.tokenizer.encode("<pad>")[:-1])
        input_length = len(input_ids)

        input_tensor = torch.zeros(batch_size, input_length).long()
        input_tensor[:] = torch.tensor(input_ids)
        finished = torch.zeros(batch_size, input_length).byte().to(device)

        for i in range(4):
            logits = self.t5_model(encoder_outputs=encoder_output,decoder_input_ids=input_tensor.to(device)).logits

            logits = F.softmax(logits[:, -1, :])
            last_token_id = torch.multinomial(logits, 1)

            EOS_sampled = (last_token_id == self.tokenizer.eos_token_id)
            finished = torch.ge(finished + EOS_sampled, 1)
            if torch.prod(finished) == 1:
                print('End')
                break
            input_tensor = torch.cat((input_tensor, last_token_id.detach().to('cpu')), 1)
            
        insert_ids = self.tokenizer.encode('G F Y A F', add_special_tokens=False)
        insert_tensor = torch.tensor(insert_ids).repeat(batch_size).reshape(batch_size, -1)
        input_tensor = torch.cat((input_tensor, insert_tensor), 1)
        
        for i in range(1):
            logits = self.t5_model(encoder_outputs=encoder_output,decoder_input_ids=input_tensor.to(device)).logits

            logits = F.softmax(logits[:, -1, :])
            last_token_id = torch.multinomial(logits, 1)

            EOS_sampled = (last_token_id == self.tokenizer.eos_token_id)
            finished = torch.ge(finished + EOS_sampled, 1)
            if torch.prod(finished) == 1:
                print('End')
                break

            input_tensor = torch.cat((input_tensor, last_token_id.detach().to('cpu')), 1)
            
        return self.tokenizer.batch_decode(input_tensor, skip_special_tokens=True)

    def gen_cdr3_89AF(self,latents,device):
        batch_size = latents.size()[0]
        encoder_output = BaseModelOutput(last_hidden_state=self.t5_model.get_decoder_input(latents.clone()))
        max_length = 10
        input_ids = []
        input_ids.extend(self.tokenizer.encode("<pad>")[:-1])
        input_length = len(input_ids)

        input_tensor = torch.zeros(batch_size, input_length).long()
        input_tensor[:] = torch.tensor(input_ids)
        finished = torch.zeros(batch_size, input_length).byte().to(device)

        for i in range(7):
            logits = self.t5_model(encoder_outputs=encoder_output,decoder_input_ids=input_tensor.to(device)).logits

            logits = F.softmax(logits[:, -1, :])
            last_token_id = torch.multinomial(logits, 1)

            EOS_sampled = (last_token_id == self.tokenizer.eos_token_id)
            finished = torch.ge(finished + EOS_sampled, 1)
            if torch.prod(finished) == 1:
                print('End')
                break
            input_tensor = torch.cat((input_tensor, last_token_id.detach().to('cpu')), 1)
            
        insert_ids = self.tokenizer.encode('A F', add_special_tokens=False)
        insert_tensor = torch.tensor(insert_ids).repeat(batch_size).reshape(batch_size, -1)
        input_tensor = torch.cat((input_tensor, insert_tensor), 1)
        
        for i in range(1):
            logits = self.t5_model(encoder_outputs=encoder_output,decoder_input_ids=input_tensor.to(device)).logits
            logits = F.softmax(logits[:, -1, :])
            last_token_id = torch.multinomial(logits, 1)

            EOS_sampled = (last_token_id == self.tokenizer.eos_token_id)
            finished = torch.ge(finished + EOS_sampled, 1)
            if torch.prod(finished) == 1:
                print('End')
                break

            input_tensor = torch.cat((input_tensor, last_token_id.detach().to('cpu')), 1)
            
        return self.tokenizer.batch_decode(input_tensor, skip_special_tokens=True)
    
    def gen_cdr3_8910AFD(self,latents,device):
        batch_size = latents.size()[0]
        encoder_output = BaseModelOutput(last_hidden_state=self.t5_model.get_decoder_input(latents.clone()))
        max_length = 10
        input_ids = []
        input_ids.extend(self.tokenizer.encode("<pad>")[:-1])
        input_length = len(input_ids)

        input_tensor = torch.zeros(batch_size, input_length).long()
        input_tensor[:] = torch.tensor(input_ids)
        finished = torch.zeros(batch_size, input_length).byte().to(device)

        for i in range(7):
            logits = self.t5_model(encoder_outputs=encoder_output,decoder_input_ids=input_tensor.to(device)).logits
            logits = F.softmax(logits[:, -1, :])
            last_token_id = torch.multinomial(logits, 1)

            EOS_sampled = (last_token_id == self.tokenizer.eos_token_id)
            finished = torch.ge(finished + EOS_sampled, 1)
            if torch.prod(finished) == 1:
                print('End')
                break
            input_tensor = torch.cat((input_tensor, last_token_id.detach().to('cpu')), 1)
            
        insert_ids = self.tokenizer.encode('A F D', add_special_tokens=False)
        insert_tensor = torch.tensor(insert_ids).repeat(batch_size).reshape(batch_size, -1)
        input_tensor = torch.cat((input_tensor, insert_tensor), 1)
        return self.tokenizer.batch_decode(input_tensor, skip_special_tokens=True)
    

    @torch.no_grad()
    def general_sample(self, fix,num_samples, sampling_timesteps=250,seed=42, class_id=None, test=False, cls_free_guidance=1.0):
        data = {'AASeq': []}
        self.ema.ema_model.eval()
        torch.manual_seed(seed)
        device = self.accelerator.device
        
        def get_class_id(n):
            if exists(class_id):
                return torch.tensor([class_id]*n, dtype=torch.long, device=device)
            if self.class_conditional:
                if self.diffusion.diffusion_model.class_unconditional_prob > 0:
                    return torch.tensor([self.diffusion.diffusion_model.num_classes]*n, dtype=torch.long, device=device)
                return self.class_categorical.sample((n,)).to(device)
            return None

        self.ema.ema_model.sampling_timesteps = sampling_timesteps
        text = []
        import time
        time_cost = 0
        while len(text) < num_samples:
            batches = num_to_groups(num_samples-len(text), self.eval_batch_size)
            start = time.time()
            model_outputs = list(map(lambda n: tuple(x.to('cpu') for x in self.ema.ema_model.sample(batch_size=n, length=self.length_categorical.sample((n,)), class_id=get_class_id(n), cls_free_guidance=cls_free_guidance)), batches))
            end = time.time()
            
            time_cost += end-start
            for (latents, mask) in model_outputs:
                latents, mask = latents.to(device), mask.to(device)
                if self.args.normalize_latent:
                    latents = self.ema.ema_model.unnormalize_latent(latents)

                if fix ==None:
                    AAs_ls = self.gen_cdr3(latents,device)
                elif fix =="12WG":
                    AAs_ls = self.gen_cdr3_12WG(latents,device)
                elif fix =="12YG":
                    AAs_ls = self.gen_cdr3_12YG(latents,device)
                elif fix =="123WGG":
                    AAs_ls = self.gen_cdr3_123WGG(latents,device)
                elif fix =="4567DGFY":
                    AAs_ls = self.gen_cdr3_4567DGFY(latents,device)
                elif fix =="567GFY":
                    AAs_ls = self.gen_cdr3_567GFY(latents,device)
                elif fix =="56789GFYAF":
                    AAs_ls = self.gen_cdr3_56789GFYAF(latents,device)
                elif fix =="89AF":
                    AAs_ls = self.gen_cdr3_89AF(latents,device)
                elif fix =="8910AFD":
                    AAs_ls = self.gen_cdr3_8910AFD(latents,device)
                text.extend(AAs_ls)
        data['AASeq'].extend(text)
        
        save_path = os.path.join(self.results_folder, f'fix{fix}_sample{num_samples}_seed{seed}_time{sampling_timesteps}.csv')

        with open(save_path, "w") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(data.keys())
            writer.writerows(zip(*data.values()))

    @torch.no_grad()
    def guided_sample(self,cond_fn, num_samples, seed=42, class_id=None, test=False, cls_free_guidance=1.0):
        data = {'AAs': []}
        self.ema.ema_model.eval()
        torch.manual_seed(seed)
        device = self.accelerator.device

        def get_class_id(n):
            if exists(class_id):
                return torch.tensor([class_id]*n, dtype=torch.long, device=device)
            if self.class_conditional:
                if self.diffusion.diffusion_model.class_unconditional_prob > 0:
                    return torch.tensor([self.diffusion.diffusion_model.num_classes]*n, dtype=torch.long, device=device)
                return self.class_categorical.sample((n,)).to(device)
            return None
        
        text = []
        while len(text) < num_samples:
            batches = num_to_groups(num_samples-len(text), self.eval_batch_size)
            model_outputs = list(map(lambda n: tuple(x.to('cpu') for x in self.ema.ema_model.sample(batch_size=n, length=self.length_categorical.sample((n,)), class_id=get_class_id(n), cls_free_guidance=cls_free_guidance)), batches))
            for (latents, mask) in model_outputs:
                latents, mask = latents.to(device), mask.to(device)
                if self.args.normalize_latent:
                    latents = self.ema.ema_model.unnormalize_latent(latents)
                AAs_ls = self.gen_cdr3(latents,device)
                text.extend(AAs_ls)
        data['AAs'].extend(text)
        
        save_path = os.path.join(self.results_folder, f'guided_sample{num_samples}_seed{seed}_pos_50.csv')

        with open(save_path, "w") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(data.keys())
            writer.writerows(zip(*data.values()))

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        best_loss = 999
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                #TODO center and normalize BART latent space with empirical est. of mean/var.
                total_loss = 0.
                for grad_accum_step in range(self.gradient_accumulate_every):
                    data = next(self.data_iter)
                    with torch.no_grad():
                        encoder_outputs = self.t5_model.get_encoder()(input_ids = data['input_ids'], attention_mask = data['attention_mask'])
                        if self.using_latent_model:
                            latent = self.t5_model.get_diffusion_latent(encoder_outputs, data['attention_mask'])      
                        else:                      
                            latent = encoder_outputs.last_hidden_state

                        if self.args.normalize_latent:
                            if self.step==0 and grad_accum_step==0:
                                if self.using_latent_model:
                                    latent_vecs = rearrange(latent, 'b s d -> (b s) d')
                                else:
                                    latent_vecs = torch.cat([latent[i][:torch.sum(data['attention_mask'][i])] for i in range(latent.shape[0])], dim=0)

                                # Add mean stats to model and EMA wrapper
                                self.diffusion.latent_mean = torch.mean(latent_vecs, dim=0)
                                self.ema.ema_model.latent_mean = self.diffusion.latent_mean

                                # Add var stats to model and EMA wrapper
                                self.diffusion.latent_scale = torch.std(latent_vecs-self.diffusion.latent_mean, unbiased=False)

                                self.ema.ema_model.latent_scale = self.diffusion.latent_scale
                            latent = self.diffusion.normalize_latent(latent)

                    if self.using_latent_model:
                        mask = torch.ones((latent.shape[0], self.num_encoder_latents), dtype=torch.bool).to(device)
                    else:
                        mask = data['attention_mask'].bool()

                    with self.accelerator.autocast():
                        loss = self.diffusion(latent, mask, class_id=(data['Label'] if self.diffusion.diffusion_model.class_conditional else None))
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.wait_for_everyone()

                grad_norm = compute_grad_norm(self.diffusion.parameters())

                accelerator.clip_grad_norm_(self.diffusion.parameters(), 1.0)
                if not self.args.resume_training:
                    self.opt.step()
                    self.lr_scheduler.step()
                    self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    # Log to WandB
                    if self.step % 100 == 0:
                        self.diffusion.eval()
                        self.ema.ema_model.eval()
                        with torch.no_grad():
                            total_val_loss = 0.
                            total_val_ema_loss = 0.
                            for grad_accum_step in range(10):
                                data = next(self.val_iter)
                                encoder_outputs = self.t5_model.get_encoder()(input_ids = data['input_ids'], attention_mask = data['attention_mask'])

                                if self.using_latent_model:
                                    latent = self.t5_model.get_diffusion_latent(encoder_outputs, data['attention_mask'])      
                                else:                      
                                    latent = encoder_outputs.last_hidden_state

                                if self.args.normalize_latent:
                                    latent = self.diffusion.normalize_latent(latent)
                            
                                if self.using_latent_model:
                                    mask = torch.ones((latent.shape[0], self.num_encoder_latents), dtype=torch.bool).to(device)
                                else:
                                    mask = data['attention_mask'].bool()

                                with self.accelerator.autocast():
                                    loss = self.diffusion(latent, mask, class_id=(data['Label'] if self.diffusion.diffusion_model.class_conditional else None))
                                    loss = loss / 10
                                    total_val_loss += loss.item()
                                    loss = self.ema.ema_model(latent, mask, class_id=(data['Label'] if self.diffusion.diffusion_model.class_conditional else None))
                                    loss = loss / 10
                                    total_val_ema_loss += loss.item()

                            if total_val_loss < best_loss:
                                best_loss = total_val_loss
                                self.save(best=True)

                            logs = {"loss": total_loss, "val_loss": total_val_loss, "val_ema_loss": total_val_ema_loss, "grad_norm": grad_norm, "lr": self.lr_scheduler.get_last_lr()[0], "step": self.step, "epoch": (self.step*self.gradient_accumulate_every)/len(self.dataloader), "samples": self.step*self.train_batch_size*self.gradient_accumulate_every}
                            pbar.set_postfix(**logs)
                            accelerator.log(logs, step=self.step)

                        self.diffusion.train()           

                    if self.step % self.save_and_sample_every == 0:
                        self.save()
                        self.diffusion.train() 
                pbar.update(1)
            accelerator.wait_for_everyone()
        self.save()
        accelerator.print('training complete')