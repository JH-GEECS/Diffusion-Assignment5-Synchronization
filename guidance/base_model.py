import os 
import torch
from abc import *
from pathlib import Path
from datetime import datetime

from diffusers import StableDiffusionPipeline, DDIMScheduler, DiffusionPipeline

from diffusion.stable_diffusion import StableDiffusion

import random
def get_current_time():
    now = datetime.now().strftime("%m-%d-%H%M%S")
    
    # to handle parallel io
    # 나중에 꼭 지우기    
    random_number = random.randint(0, 9999)
    random_number_str = f"{random_number:04d}"
    now = f"{now}_{random_number_str}"
    
    return now

    
class BaseModel(metaclass=ABCMeta):
    def __init__(self):
        self.init_model()
        self.init_mapper()
        
    def initialize(self):
        now = get_current_time()
        save_top_dir = self.config.save_top_dir
        tag = self.config.tag
        save_dir_now = self.config.save_dir_now 
        
        if save_dir_now:
            self.output_dir = Path(save_top_dir) / f"{tag}/{now}"
        else:
            self.output_dir = Path(save_top_dir) / f"{tag}"
        
        if not os.path.isdir(self.output_dir):
            self.output_dir.mkdir(exist_ok=True, parents=True)
        else:
            print(f"Results exist in the output directory, use time string to avoid name collision.")
            exit(0)
            
        print("[*] Saving at ", self.output_dir)
    
    
    @abstractmethod
    def init_mapper(self, **kwargs):
        pass
    
    
    @abstractmethod
    def forward_mapping(self, z_t, **kwargs):
        pass
    
    
    @abstractmethod
    def inverse_mapping(self, x_ts, **kwargs):
        pass
    
    
    @abstractmethod
    def compute_noise_preds(self, xts, ts, **kwargs):
        pass
        
    
    def init_model(self):
        if self.config.model == "sd":
            pipe = StableDiffusionPipeline.from_pretrained(
                self.config.sd_path,
                torch_dtype=torch.float16,
            ).to(self.device)
            
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            self.model = StableDiffusion(
                **pipe.components,
            )
            
            del pipe
            
        elif self.config.model == "deepfloyd":
            self.stage_1 = DiffusionPipeline.from_pretrained(
                "DeepFloyd/IF-I-M-v1.0", 
                variant="fp16", 
                torch_dtype=torch.float16,
            )
            self.stage_2 = DiffusionPipeline.from_pretrained(
                "DeepFloyd/IF-II-M-v1.0",
                text_encoder=None,
                variant="fp16",
                torch_dtype=torch.float16,
            )
            
            scheduler = DDIMScheduler.from_config(self.stage_1.scheduler.config)
            self.stage_1.scheduler = self.stage_2.scheduler = scheduler
            
        else:
            raise NotImplementedError(f"Invalid model: {self.config.model}")
        
        
        if self.config.model in ["sd"]:
            self.model.text_encoder.requires_grad_(False)
            self.model.unet.requires_grad_(False)
            if hasattr(self.model, "vae"):
                self.model.vae.requires_grad_(False)
        else:
            self.stage_1.text_encoder.requires_grad_(False)
            self.stage_2.unet.requires_grad_(False)
            self.stage_2.unet.requires_grad_(False)
            
            self.stage_1 = self.stage_1.to(self.device)
            self.stage_2 = self.stage_2.to(self.device)
                
                
    def compute_tweedie(self, xts, eps, timestep, alphas, sigmas, **kwargs):
        """
        Input:
            xts, eps: [B,*]
            timestep: [B]
            x_t = alpha_t * x0 + sigma_t * eps
            assume that alphas is sqrted. checkout wide_image_model.py/WideImageModel/__call__
        Output:
            pred_x0s: [B,*]
        """
        
        # import ipdb; ipdb.set_trace()
        # print("xts: ", xts.shape) [2,3,64,64]
        # print("eps: ", eps.shape) [2,6,64,64]
        # print("timestep: ", timestep.shape)
        # print("alphas: ", alphas.shape)
        
        if self.config.model == "deepfloyd":
            eps, _ = torch.split(eps, xts.shape[-3], dim=-3)
        
        
        pred_x0s = (xts - ((1- alphas[timestep] ** 2) ** 0.5) * eps) / (alphas[timestep])
        
        # TODO: Task 0, Implement compute_tweedie
        # raise NotImplementedError("compute_tweedie is not implemented yet.")
        return pred_x0s
        
    def compute_prev_state(
        self, xts, pred_x0s, timestep, **kwargs,
    ):
        """
        Input:
            xts: [N,C,H,W]
        ... how to compute alphas?
        
        Output:
            pred_prev_sample: [N,C,H,W]
        """
        
        if self.config.model == "sd":
            scheduler: DDIMScheduler = self.model.scheduler
        elif self.config.model == "deepfloyd":
            scheduler: DDIMScheduler = self.stage_1.scheduler
        else:
            raise NotImplementedError(f"Invalid model: {self.config.model}")
        
        alpha_t = scheduler.alphas_cumprod[timestep]
        interval = scheduler.config.num_train_timesteps // scheduler.num_inference_steps
        prev_time_step = torch.where(timestep - interval <= 0, torch.tensor(0), timestep - interval)
        alpha_prev_t = scheduler.alphas_cumprod[prev_time_step]
        
        x_prev_t = (alpha_prev_t ** 0.5) * pred_x0s +  (((1 - alpha_prev_t)/(1 - alpha_t)) ** 0.5) * (xts - (alpha_t ** 0.5) * pred_x0s)
        
        # x_prev_t = (((1 - alpha_prev_t)/(1 - alpha_t)) ** 0.5) * xts + (alpha_prev_t ** 0.5 - ((alpha_t) / (1 - alpha_t)) ** 0.5) * pred_x0s
        
        # TODO: Task 0, Implement compute_prev_state
        # raise NotImplementedError("compute_prev_state is not implemented yet.")
        return x_prev_t        
        
    def one_step_process(
        self, input_params, timestep, alphas, sigmas, **kwargs
    ):
        """
        Input:
            latents: either xt or zt. [B,*]
        Output:
            output: the same with latent.
        """
        
        xts = input_params["xts"]

        eps_preds = self.compute_noise_preds(xts, timestep, **kwargs)
        
        # sync on \epsilon_{\theta}, case 1 in sync tweedies
        
        x0s = self.compute_tweedie(
            xts, eps_preds, timestep, alphas, sigmas, **kwargs
        )
        
        # Synchronization using SyncTweedies 
        # sync on x_0, case 2 in sync tweedies
        z0s = self.inverse_mapping(x0s, var_type="tweedie", **kwargs) # Comment out to skip synchronization
        x0s = self.forward_mapping(z0s, bg=x0s, **kwargs) # Comment out to skip synchronization
        
        x_t_1 = self.compute_prev_state(xts, x0s, timestep, **kwargs)
        
        # sync on x_{t-1}, case 3 in sync tweedies

        out_params = {
            "x0s": x0s,
            "z0s": None,
            "x_t_1": x_t_1,
            "z_t_1": None,
        }

        return out_params