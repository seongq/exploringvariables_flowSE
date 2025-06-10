import time
from math import ceil
import warnings
import numpy as np
import torch
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage
import torch.nn.functional as F
from flowmse import sampling
from flowmse.odes import ODERegistry
from flowmse.backbones import BackboneRegistry
from flowmse.util.inference import evaluate_model
from flowmse.util.other import pad_spec
import numpy as np
import matplotlib.pyplot as plt
# from flowmse.odes import OTFLOW
import random



class VFModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum time (0 by default)")
        parser.add_argument("--T_rev",type=float, default=1.0, help="The maximum time")
        
        parser.add_argument("--num_eval_files", type=int, default=10, help="Number of files for speech enhancement performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).")
        parser.add_argument("--loss_type", type=str, default="mse", help="The type of loss function to use.")
        parser.add_argument("--loss_abs_exponent", type=float, default= 0.5,  help="magnitude transformation in the loss term")
        parser.add_argument("--mode_", type=str, required=True, choices=("noisemean_direct_estimation_xt_y_t", "CTFSE_KDfromclean", "flowse_KD_enindg_without_t_sequential_update", "flowse_KDfromclean", "noisemean_xt_y_sigmaz_t", "noisemean_xt_y_yplussigmaz", "noisemean_t_y","noisemean_xt_y_sigmaz", "noisemean_t_times_y_plus_sigmaz_1minust_times_s", "noisemean_xt_y_t", "noisemean_xtplusy_divide_2", "noisemean_xt_y_plus_sigmaz", "noisemean_xt_y","noisemean_conditionfalse_timefalse", "noisemean_noxt_conditiony_timefalse","noisemean_y_plus_sigmaz","noisemean_xt_t"))
        return parser
    """
    model.step, inference, evaluation code 바꿀것
    mode_
    noisemean_conditionfalse_timefalse : v_theta(x_t)
    noisemean_noxt_conditiony_timefalse : v_theta(y)
    noisemean_y_plus_sigmaz : v_theta(y+sigma z)
    noisemean_xt_y: v_theta(xt,y)
    noisemean_xt_y_plus_sigmaz: v_theta(xt,y+sigma z)   
    noisemean_xt_y_t: v_theta(xt,y,t)
    noisemean_xt_t: v_theta(xt,t)
    noisemean_t_times_y_plus_sigmaz_1minust_times_s: v_theta(t(y+sigma z), (1-t)s)
    
    
    
    """

    def __init__(
        self, backbone, ode, lr=1e-4, ema_decay=0.999, t_eps=0.03, T_rev = 1.0,  loss_abs_exponent=0.5, 
        num_eval_files=10, loss_type='mse', data_module_cls=None, N_enh=10, mode_="noisemean_conditionfalse_timefalse", **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: Backbone DNN that serves as a score-based model.
            sde: The SDE that defines the diffusion process.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            loss_type: The type of loss to use (wrt. noise z/std). Options are 'mse' (default), 'mae'
        """
        super().__init__()
        # Initialize Backbone DNN
        self.mode_ = mode_
        if self.mode_ == "noisemean_conditionfalse_timefalse":
            kwargs.update(num_channels=2)
            kwargs.update(conditional=False)
        elif self.mode_ == "noisemean_direct_estimation_xt_y_t":
            kwargs.update(num_channels=4)
            kwargs.update(conditional=True)   
        
        elif self.mode_ == "noisemean_noxt_conditiony_timefalse":
            kwargs.update(num_channels=2)
            kwargs.update(conditional=False)
        elif self.mode_ == "noisemean_y_plus_sigmaz":
            kwargs.update(num_channels=2)
            kwargs.update(conditional=False)            
        elif self.mode_ == "noisemean_xt_t": #noisemean_xt_t: v_theta(xt,t)
            kwargs.update(num_channels=2)
            kwargs.update(conditional=True)
        elif self.mode_ == "noisemean_xt_y": #noisemean_xt_y: v_theta(xt,y)
            kwargs.update(num_channels=4)
            kwargs.update(conditional=False) 
        elif self.mode_ == "noisemean_xt_y_plus_sigmaz": #v_theta(xt,y+sigma z)
            kwargs.update(num_channels=4)
            kwargs.update(conditional=False)
        elif self.mode_ == "noisemean_xtplusy_divide_2": #v_theta((xt+y)/2)
            kwargs.update(num_channels=2)
            kwargs.update(conditional=False)
        elif self.mode_ == "noisemean_xt_y_t": # noisemean_xt_y_t v_theta (xt,y,t)
            kwargs.update(num_channels=4)
            kwargs.update(conditional=True)
        elif self.mode_ == "noisemean_xt_t: v_theta(xt,t)": #noisemean_xt_t: v_theta (xt,t)
            kwargs.update(num_channels=2)
            kwargs.update(conditional=True)
        elif self.mode_ == "noisemean_t_times_y_plus_sigmaz_1minust_times_s": #": v_theta(t(y+sigma z), (1-t)s)"
            kwargs.update(num_channels=4)
            kwargs.update(conditional=False)
        elif self.mode_ == "noisemean_xt_y_sigmaz": #v_theta(xt,y,sigmaz)
            kwargs.update(num_channels=6)
            kwargs.update(conditional=False)
        elif self.mode_ == "noisemean_t_y": #v_theta(t,y)
            kwargs.update(num_channels=2)
            kwargs.update(conditional=True)
        elif self.mode_ == "noisemean_xt_y_yplussigmaz": #v_theta(xt,y,y+sigma z)
            kwargs.update(num_channels=6)
            kwargs.update(conditional=False)
        elif self.mode_ == "noisemean_xt_y_sigmaz_t": #v_theta(t,xt,y,sigmaz)
            kwargs.update(num_channels=6)
            kwargs.update(conditional=True)
        elif self.mode_ == "noisemean_xt_y_sigmaz_yplussigmaz_t": #v_theta(t,xt,y,y+sigmaz, sigmaz)
            kwargs.update(num_channels=8)
            kwargs.update(conditional=True)
        elif self.mode_ =="flowse_KDfromclean":
            kwargs.update(num_channels=4)
            kwargs.update(conditional=True)
        elif self.mode_ =="flowse_KD_enindg_without_t_sequential_update":
            kwargs.update(num_channels=4)
            kwargs.update(conditional=False)
            self.automatic_optimization = False  # ★ 중요 ★
        elif self.mode_ =="CTFSE_KDfromclean":
            kwargs.update(num_channels=4)
            kwargs.update(conditional=True)
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        self.dnn = dnn_cls(**kwargs)        
        # for name, param in self.dnn.named_parameters():
        #     print(name, param.requires_grad)
        ode_cls = ODERegistry.get_by_name(ode)
        # self.mode_condition = mode_condition
       
        
        
        self.ode = ode_cls(**kwargs)
        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.T_rev = T_rev
        self.ode.T_rev = T_rev
        self.loss_type = loss_type
        self.num_eval_files = num_eval_files
        self.loss_abs_exponent = loss_abs_exponent
        self.save_hyperparameters(ignore=['no_wandb'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)
        # self.mode = mode


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.parameters())        # store current params in EMA
                self.ema.copy_to(self.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _mse_loss(self, x, x_hat):    
        err = x-x_hat
        losses = torch.square(err.abs())

        # taken from reduce_op function: sum over channels and position and mean over batch dim
        # presumably only important for absolute loss number, not for gradients
        loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss
    
    
    def _loss(self, vectorfield, condVF):    
        if self.loss_type == 'mse':
            err = vectorfield-condVF
            losses = torch.square(err.abs())
        elif self.loss_type == 'mae':
            err = vectorfield-condVF
            losses = err.abs()
        # taken from reduce_op function: sum over channels and position and mean over batch dim
        # presumably only important for absolute loss number, not for gradients
        loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss

    def _step(self, batch, batch_idx, step_num = 0):
        import random
        x0, y = batch
        rdm = (1-torch.rand(x0.shape[0], device=x0.device)) * (self.T_rev-self.t_eps) +self.t_eps
        t = torch.min(rdm, torch.tensor(self.T_rev))
        mean, std = self.ode.marginal_prob(x0, t, y)
        z = torch.randn_like(x0)  #
        SIGMA = self.ode._std(1)
        # print(SIGMA)
        sigmas = std[:, None, None, None]
        xt = mean + sigmas * z
        der_std = self.ode.der_std(t)
        der_mean = self.ode.der_mean(x0,t,y)
        condVF = der_std * z + der_mean
        
        
        
        if self.mode_ == "noisemean_conditionfalse_timefalse":
            VECTORFIELD_origin = self(t,xt) 
        elif self.mode_ =="CTFSE_KDfromclean":
            
            s = x0
            y = y  
            
            rdm = (1-torch.rand(x0.shape[0], device=x0.device)) * (self.T_rev - self.t_eps) + self.t_eps        
            t = torch.min(rdm, torch.tensor(self.T_rev))
            mean, std = self.ode.marginal_prob(s, t, y)
            z = torch.randn_like(s)  #
            sigmas = std[:, None, None, None]
            xt = mean + sigmas * z
            der_std = self.ode.der_std(t)
            der_mean = self.ode.der_mean(s,t,y)
            condVF = der_std * z + der_mean    
            VECTORFIELD_origin = self(t,xt,y)
            
            with torch.no_grad():
                VECTOFIELD_origin_teacher = self(t,xt,s)
            
            
            loss_original_flow = self._loss(VECTORFIELD_origin,condVF)
            loss_original_flow_kd = self._loss(VECTORFIELD_origin, VECTOFIELD_origin_teacher)
            
            
              
            # print("none")
            
            x1, _ = self.ode.prior_sampling(y.shape,y)
            ENHANCER = self(torch.ones(y.shape[0], device=y.device), x1,y)
            ENHANCEMENT = x1 - ENHANCER
            
            loss_enh = self._loss(ENHANCEMENT,s)
            
            CONDITION = 0.5 * ENHANCEMENT + 0.5 * y
            rdm = (1-torch.rand(x0.shape[0], device=x0.device)) * (self.T_rev - self.t_eps) + self.t_eps        
            t = torch.min(rdm, torch.tensor(self.T_rev))
            mean, std = self.ode.marginal_prob(s, t, ENHANCEMENT)
            z = torch.randn_like(s)  #
            sigmas = std[:, None, None, None]
            xt = mean + sigmas * z
            der_std = self.ode.der_std(t)
            der_mean = self.ode.der_mean(s,t,ENHANCEMENT)
            condVF = der_std * z + der_mean    
            VECTORFIELD = self(t,xt,CONDITION)
            loss_flow = self._loss(VECTORFIELD,condVF)
            with torch.no_grad():
                VECTORFIELD_teacher = self(t,xt,s)
            loss_flow_kd = self._loss(VECTORFIELD,VECTORFIELD_teacher )
            
            
            loss = loss_original_flow + loss_flow + loss_enh + loss_flow_kd + loss_original_flow_kd
            return loss
        elif self.mode_ == "noisemean_noxt_conditiony_timefalse":
            VECTORFIELD_origin = self(t,y)
        elif self.mode_ == "noisemean_y_plus_sigmaz":
            VECTORFIELD_origin = self(t,y+sigmas * z)
        elif self.mode_ == "noisemean_xt_t": #noisemean_xt_t: v_theta(xt,t)
            VECTORFIELD_origin = self(t,xt)
        elif self.mode_ == "noisemean_xt_y": #noisemean_xt_y: v_theta(xt,y)
            VECTORFIELD_origin = self(t,xt,y)
        elif self.mode_ == "noisemean_xt_y_plus_sigmaz": #v_theta(xt,y+sigma z)
            VECTORFIELD_origin = self(t,xt,y+SIGMA * z)
        elif self.mode_ == "noisemean_xtplusy_divide_2": #v_theta((xt+y)/2)
            VECTORFIELD_origin = self(t,(xt+y)/2)
        elif self.mode_ == "noisemean_xt_y_t": # noisemean_xt_y_t v_theta (xt,y,t)
            VECTORFIELD_origin = self(t,xt,y)
        elif self.mode_ == "noisemean_xt_t: v_theta(xt,t)": #noisemean_xt_t: v_theta(xt,t) v_\theta(xt,t)
            VECTORFIELD_origin = self(t,xt)
        elif self.mode_ == "noisemean_t_times_y_plus_sigmaz_1minust_times_s": #": v_theta(t(y+sigma z), (1-t)s)"
            s = x0
            vect = torch.ones(y.shape[0], device=y.device) * t
            vect = vect[:,None,None,None]
            
            # print(t.shape)
            # t = t[:, None, None, None]
            # print(t.shape)
            # print(vect.shape)
            # print(y.shape)
            # print(z.shape)
            # print(s.shape)
            VECTORFIELD_origin = self(t,vect*(y+SIGMA*z), (1-vect)*s)
            
        elif (self.mode_ == "noisemean_xt_y_sigmaz") or (self.mode_ == "noisemean_xt_y_sigmaz_t"): #v_theta(t,xt,y,sigmaz)
            VECTORFIELD_origin = self(t,xt,y, SIGMA * z)
            
        elif self.mode_ == "noisemean_t_y": #v_theta(t,y)
            VECTORFIELD_origin  = self(t,y)
            
        elif self.mode_ == "noisemean_xt_y_yplussigmaz": #v_theta(xt,y,y+sigma z)
            VECTORFIELD_origin = self(t, xt, y, y+ SIGMA *z )
        elif self.mode_ == "noisemean_direct_estimation_xt_y_t": #s_theta(t,xt,y)
            clean_estimated = self(t,xt,y)
            loss = self._loss(clean_estimated, x0)
            return loss
        
        elif self.mode_ == "noisemean_xt_y_sigmaz_yplussigmaz_t": #v_theta(t,xt,y,y+sigmaz, sigmaz)
            VECTORFIELD_origin = self(t,xt,y,y+SIGMA*z, SIGMA*z)
        elif self.mode_ == "flowse_KDfromclean":
            VECTORFIELD_origin = self(t,xt,y)
            with torch.no_grad():
                VECTORFIELD_CLEAN = self(t,xt,x0)
            loss_original_flow = self._loss(VECTORFIELD_origin, condVF)
            loss_clean = self._loss(VECTORFIELD_CLEAN,VECTORFIELD_origin)
            loss = loss_original_flow + loss_clean
            
            return loss
                
        elif self.mode_ =="flowse_KD_enindg_without_t_sequential_update":
            # print(step_num)
            x0, y = batch
            
            
            if self.training:
                if step_num == 0:
                    rdm = (1-torch.rand(x0.shape[0], device=x0.device)) * (self.T_rev-self.t_eps) +self.t_eps
                    t = torch.min(rdm, torch.tensor(self.T_rev))
                    mean, std = self.ode.marginal_prob(x0, t, y)
                    z = torch.randn_like(x0)  #
                    # SIGMA = self.ode._std(1)
                    # print(SIGMA)
                    sigmas = std[:, None, None, None]
                    xt = mean + sigmas * z
                    der_std = self.ode.der_std(t)
                    der_mean = self.ode.der_mean(x0,t,y)
                    condVF = der_std * z + der_mean

                
                    
                    VECTORFIELD_t = self(t,xt,y)
                    
                    condVF_t = condVF
                    loss = self._loss(VECTORFIELD_t, condVF_t)
                    
                
                
                
                elif step_num == 1:
                    
                
                    rdm = (1-torch.rand(x0.shape[0], device=x0.device)) * torch.tensor(0) +self.t_eps
                    t_eps = torch.min(rdm, torch.tensor(self.T_rev))
                    mean, std = self.ode.marginal_prob(x0, t_eps, y)
                    
                    SIGMA = self.ode._std(1)
                    # print(SIGMA)
                    sigmas = std[:, None, None, None]
                    xt_eps = mean + sigmas * z
                    der_std = self.ode.der_std(t_eps)
                    der_mean = self.ode.der_mean(x0,t_eps,y)
                    condVF_t_eps = der_std * z + der_mean
                    
                    VECTORFIELD_t_eps = self(t_eps,xt_eps,y)
                    loss = self._loss(VECTORFIELD_t_eps, condVF_t_eps)
                    
                    
                
                elif step_num == 2:
                
                    rdm = (1-torch.rand(x0.shape[0], device=x0.device)) * torch.tensor(0) +self.T_rev
                    t_1 = torch.min(rdm, torch.tensor(self.T_rev))
                    mean, std = self.ode.marginal_prob(x0, t_1, y)
                    # SIGMA = self.ode._std(1)
                    # print(SIGMA)
                    sigmas = std[:, None, None, None]
                    xt_1 = mean + sigmas * z
                    der_std = self.ode.der_std(t_1)
                    der_mean = self.ode.der_mean(x0,t_1,y)
                    condVF = der_std * z + der_mean
                    
                    VECTORFIELD_t_1 = self(t_1,xt_1,y)
                    loss = self._loss(VECTORFIELD_t_1, condVF)
                
                
                elif step_num ==3:
                    rdm = (1-torch.rand(x0.shape[0], device=x0.device)) * (self.T_rev-self.t_eps) +self.t_eps
                    z = torch.randn_like(x0)  #
                    
                    
                    
                    t = torch.min(rdm, torch.tensor(self.T_rev))
                    mean, std = self.ode.marginal_prob(x0, t, y)
                    
                    sigmas = std[:, None, None, None]
                    xt = mean + sigmas * z
                    
                    VECTORFIELD_t = self(t,xt,y)
                    
                    
                    
                    
                    t_eps = self.t_eps
                
                    rdm = (1-torch.rand(x0.shape[0], device=x0.device)) * torch.tensor(0) +self.t_eps
                    t_eps = torch.min(rdm, torch.tensor(self.T_rev))
                    mean, std = self.ode.marginal_prob(x0, t_eps, y)
                    
                    sigmas = std[:, None, None, None]
                    xt_eps = mean + sigmas * z
                    
                    
                    VECTORFIELD_t_eps = self(t_eps,xt_eps,y).detach()
                    
                    
                    
                    
                    rdm = (1-torch.rand(x0.shape[0], device=x0.device)) * torch.tensor(0) +self.T_rev
                    t_1 = torch.min(rdm, torch.tensor(self.T_rev))
                    mean, std = self.ode.marginal_prob(x0, t_1, y)
                    
                    sigmas = std[:, None, None, None]
                    xt_1 = mean + sigmas * z
                    
                    
                    VECTORFIELD_t_1 = self(t_1,xt_1,y)
                    
                    loss = self._loss(VECTORFIELD_t_eps, VECTORFIELD_t)+self._loss(VECTORFIELD_t_eps, VECTORFIELD_t_1)
            else:
                rdm = (1-torch.rand(x0.shape[0], device=x0.device)) * (self.T_rev-self.t_eps) +self.t_eps
                z = torch.randn_like(x0)  #
                
                
                
                t = torch.min(rdm, torch.tensor(self.T_rev))
                mean, std = self.ode.marginal_prob(x0, t, y)
                
                sigmas = std[:, None, None, None]
                xt = mean + sigmas * z
                
                VECTORFIELD_t = self(t,xt,y)
                
                
                
                
                t_eps = self.t_eps
               
                rdm = (1-torch.rand(x0.shape[0], device=x0.device)) * torch.tensor(0) +self.t_eps
                t_eps = torch.min(rdm, torch.tensor(self.T_rev))
                mean, std = self.ode.marginal_prob(x0, t_eps, y)
                
                sigmas = std[:, None, None, None]
                xt_eps = mean + sigmas * z
                
                
                VECTORFIELD_t_eps = self(t_eps,xt_eps,y).detach()
                
                
                
                
                rdm = (1-torch.rand(x0.shape[0], device=x0.device)) * torch.tensor(0) +self.T_rev
                t_1 = torch.min(rdm, torch.tensor(self.T_rev))
                mean, std = self.ode.marginal_prob(x0, t_1, y)
                
                sigmas = std[:, None, None, None]
                xt_1 = mean + sigmas * z
                
                
                VECTORFIELD_t_1 = self(t_1,xt_1,y)
                
                loss = self._loss(VECTORFIELD_t_eps, VECTORFIELD_t)+self._loss(VECTORFIELD_t_eps, VECTORFIELD_t_1)
            
            
            
            
            return loss
            
            
        loss_original_flow = self._loss(VECTORFIELD_origin,condVF)
        

        loss = loss_original_flow 
        return loss
    
    def training_step(self, batch, batch_idx):
        if self.mode_ == "flowse_KD_enindg_without_t_sequential_update":
            opt = self.optimizers()  # Lightning이 제공하는 optimizer 핸들
            n_inner_steps = 4        # 원하는 반복 횟수
            total_loss = 0.0

            for i in range(n_inner_steps):
                loss = self._step(batch, batch_idx, step_num=i)  # step_index는 optional
                opt.zero_grad()
                self.manual_backward(loss)
                opt.step()
                self.ema.update(self.parameters())  # 선택적

                self.log(f'train_loss_step_{i}', loss, on_step=True, prog_bar=True)
                total_loss += loss.detach()

            # self.log("train_loss", total_loss / n_inner_steps, on_step=False, on_epoch=True)
            return total_loss
        else:    
            loss = self._step(batch, batch_idx)
            self.log('train_loss', loss, on_step=True, on_epoch=True)
            return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True)

        # Evaluate speech enhancement performance
        if batch_idx == 0 and self.num_eval_files != 0:
            pesq, si_sdr, estoi = evaluate_model(self, self.num_eval_files)
            self.log('pesq', pesq, on_step=False, on_epoch=True)
            self.log('si_sdr', si_sdr, on_step=False, on_epoch=True)
            self.log('estoi', estoi, on_step=False, on_epoch=True)

        return loss

    def forward(self,t, *args):
        # Concatenate y as an extra channel
        dnn_input = torch.cat(args, dim=1)
        
        # the minus is most likely unimportant here - taken from Song's repo
        score = -self.dnn(dnn_input, t)
        return score

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)


    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)
    