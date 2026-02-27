import numpy as np
import tqdm
import torch

from guided_diffusion.script_util import sr_create_model_and_diffusion, sr_model_and_diffusion_defaults
import random
from pathlib import Path
from physics.projector import create_projector, create_fwd_proj_layer, create_back_proj_layer
from utils import clear
# adaptation
from lora.lora import adapt_model
from lora.adaptation import adapt_loss_fn

def create_sr_kwargs(config):
    sr_kwargs = dict(
        large_size                = config.model.large_size,
        small_size                = config.model.small_size,
        num_channels              = config.model.num_channels,
        num_res_blocks            = config.model.num_res_blocks,
        attention_resolutions     = config.model.attention_resolutions,
        dropout                   = config.model.dropout,
        use_fp16                  = config.model.use_fp16,
        learn_sigma               = config.model.learn_sigma,
        resblock_updown           = config.model.resblock_updown,
        use_scale_shift_norm      = config.model.use_scale_shift_norm,
        num_head_channels         = config.model.num_head_channels,
        
        diffusion_steps           = config.diffusion.num_diffusion_timesteps,
        noise_schedule            = config.diffusion.beta_schedule,
        rescale_learned_sigmas    = config.diffusion.rescale_learned_sigmas,
        rescale_timesteps         = config.diffusion.rescale_timesteps,
    )
    return sr_kwargs

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None, coeff_schedule="ddnm"):
        self.args = args
        self.coeff_schedule = coeff_schedule
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self):
        cls_fn = None
        defaults = sr_model_and_diffusion_defaults()
        config_dict = create_sr_kwargs(self.config)
        for k, v in config_dict.items():
            if k in defaults:
                if k == "timestep_respacing":
                    v = str(v)
                defaults[k] = v
            else:
                print(f"[WARN] {k} is not in sr_model_and_diffusion_defaults(), ignoring.")

        model, diffusion = sr_create_model_and_diffusion(**defaults)
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        nparams = count_parameters(model)
        print(f"Number of parameters: {nparams}")
        ckpt = self.config.model.model_ckpt
        ckpt_f = torch.load(ckpt, map_location=self.device)
        model.load_state_dict(ckpt_f)
        print(f"Model ckpt loaded from {ckpt}")
        model.to(self.device)
        model.convert_to_fp32()
        model.dtype = torch.float32
        model.eval()
        if self.args.adaptation and int(self.args.lora_rank) > 0:
            print('With LoRA')
            adapt_kwargs = {'r': int(self.args.lora_rank)}
            adapt_model(model, adapt_kwargs=adapt_kwargs)
        else:
            print('Without LoRA')

        self.adaptation = True if self.args.adaptation else False
        print('Running DDIP for PET reconstruction.',
            f'{self.args.T_sampling} sampling steps.',
            f'Task: {self.args.deg}.'
            f'Adaptation?: {self.adaptation}'
            )
        self.simplified_ddnm_plus(model)
            
            
    def simplified_ddnm_plus(self, model):
        args, config = self.args, self.config
        save_root = args.save_root
        test_root = args.test_root
        save_root = Path(f'{self.args.save_root}')
        save_root.mkdir(parents=True, exist_ok=True)
        
        # read test data
        print("Loading test data")
        x_orig = []
        y_orig = []
        sino = torch.from_numpy(np.load(test_root + '19-000_sino.npy', allow_pickle=True))
        print('Load data: ' + test_root)
        img = np.load(test_root + '19-000_mr.npy', allow_pickle=True)
        img = img / img.max()
        img = torch.from_numpy(img)
        z, h, w = torch.squeeze(img).shape
        sino = sino.view(127, 1, 129, 224, 1)
        img = img.view(z, 1, h, w)
        y_orig.append(sino)
        x_orig.append(img)
        y_orig = torch.cat(y_orig, dim=0)
        x_orig = torch.cat(x_orig, dim=0)

        proj = create_projector()
        fwd_proj_layer = create_fwd_proj_layer()
        back_proj_layer = create_back_proj_layer()
        print("img_size: {}, N_views: {}".format(h, self.args.Nview))
        A = lambda z: fwd_proj_layer(z, proj)
        AT = lambda z: back_proj_layer(z, proj)
        
        y_orig = y_orig.to(self.device)
        x_orig = x_orig.to(self.device)
        """
        Actual inference running...
        """
        output = []
        z_ = [60]
        for idx in z_:
            x = torch.randn_like(x_orig[idx-1:idx+2, ...].view(1, 3, h, w)).to(self.device)
            x0_t_hat = torch.ones_like(x_orig[idx-1:idx+2, ...].view(1, 3, h, w)).to(self.device)
            skip = (args.start_t-args.end_t)//args.T_sampling
            n = x.size(0)
            x0_preds = []
            
            # generate time schedule
            times = list(range(args.end_t, args.start_t, skip))
            times_next = [-1] + times[:-1]
            times_pair = list(zip(reversed(times), reversed(times_next)))
            y_idx = y_orig[idx-1:idx+2, ...] * 1.
            x_idx = x_orig[idx-1:idx+2, ...].view(1, 3, h, w) * 1.
            i0, j0 = times_pair[0]
            next_t = (torch.ones(n) * j0).to("cuda")
            at_next = compute_alpha(self.betas, next_t.long())

            # MLEM initialization
            x = MLEM(y_idx.view(3, 1, 129, 1, 224), x0_t_hat.view(3, 1, h, 1, w), 20, A, AT).view(1, 3, h, w)
            x = at_next.sqrt() * x + (1 - at_next).sqrt() * torch.randn_like(x)
            xs = [x]
            
            # reverse diffusion sampling
            for i, j in tqdm.tqdm(times_pair, total=len(times)):
                t = (torch.ones(n) * i).to("cuda")
                next_t = (torch.ones(n) * j).to("cuda")
                at = compute_alpha(self.betas, t.long())
                at_next = compute_alpha(self.betas, next_t.long())
                """
                Block 1: Adaptation
                """
                if args.adaptation:
                    xt = xs[-1].to('cuda')
                    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
                    for _ in range(args.num_steps):
                        optim.zero_grad()
                        et = model(xt, t, x_idx)
                        if et.size(1) == 2:
                            et = et[:, :1]
                        if et.size(1) == 6:
                            et = et[:, :3]
                        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                        # Penalized EM reconstruction
                        x0_em = penalized_EM(y_idx.view(3, 1, 129, 1, 224), x0_t.view(3, 1, h, 1, w), int(args.em_itr), A, AT, beta=args.em_beta).view(1, 3, h, w)
                        loss = adapt_loss_fn(x0_t, x0_em)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optim.step()
                """
                Block 2: Inference after adaptation
                """
                xt = xs[-1].to('cuda')
                xt = xt.detach().clone().requires_grad_(True)
                et = model(xt, t, x_idx)

                if et.size(1) == 2:
                    et = et[:, :1]
                if et.size(1) == 6:
                    et = et[:, :3]

                # 1. Tweedie
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                x0_t_hat = x0_t

                with torch.no_grad():
                    x0_t_hat = torch.clamp(x0_t_hat, 0., 1.0)

                    eta = self.args.eta
                    c1 = (1 - at_next).sqrt() * eta
                    c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)

                    # DDIM sampling
                    if j != 0:
                        xt_next = at_next.sqrt() * x0_t_hat + c1 * torch.randn_like(x0_t) + c2 * et
                    # Final step
                    else:
                        xt_next = x0_t_hat
                        
                    x0_preds.append((x0_t).to('cpu'))
                    xs.append((xt_next).to('cpu'))
                x = xs[-1]
            recon = clear(x)
            
            output.append(recon)
        torch.cuda.synchronize()
        np.save(str(save_root / f"result.npy"), np.array(output))

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def MLEM(sng, x0t, N_iter, A, AT):
    with torch.no_grad():
        eps = 1e-10
        wgt = AT(torch.ones_like(sng))
        recon = torch.clamp(x0t, min=0)
        for n in range(N_iter):
                fp = A(recon).view(3, 1, 129, 1, 224)
                ratio = sng / (fp + eps)
                bp = AT(ratio.view(3, 1, 129, 1, 224))
                recon *= bp / wgt
        return recon
    
def Poisson_grad(sng, x0t, A, AT):
    with torch.no_grad():
        eps = 1e-10
        wgt = AT(torch.ones_like(sng))
        x = torch.clamp(x0t, min=0)
        fp = A(x).view(3, 1, 129, 1, 224)
        ratio = sng / (fp + eps)
        bp = AT(ratio.view(3, 1, 129, 1, 224))
        grad = (x / (wgt + eps)) * (bp - wgt)
        return grad
    
def penalized_EM(sng, x0t, N_iter, A, AT, beta):
    with torch.no_grad():
        eps = 1e-10
        wgt = AT(torch.ones_like(sng))
        x0t = torch.clamp(x0t, min=0)
        xem = x0t
        for n in range(N_iter):
                xem = MLEM(sng, xem, 1, A, AT)
                first_term = x0t - wgt / beta
                second_term = torch.pow((x0t - wgt/beta), 2) + 4 * xem * wgt / beta
                second_term = torch.sqrt(second_term)
                xem = (first_term + second_term) / 2.
        return xem