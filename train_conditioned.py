import argparse
from argparse import ArgumentParser
import os
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
# from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping

import wandb

from flowmse.backbones.shared import BackboneRegistry
from flowmse.data_module import SpecsDataModule
from flowmse.odes import ODERegistry
from flowmse.model import VFModel

import torch
# torch.set_num_threads(5)
# torch.cuda.empty_cache()
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
def get_argparse_groups(parser):
     groups = {}
     for group in parser._action_groups:
          group_dict = { a.dest: getattr(args, a.dest, None) for a in group._group_actions }
          groups[group.title] = argparse.Namespace(**group_dict)
     return groups


if __name__ == '__main__':
     # throwaway parser for dynamic args - see https://stackoverflow.com/a/25320537/3090225
     base_parser = ArgumentParser(add_help=False)
     parser = ArgumentParser()
     for parser_ in (base_parser, parser):
          parser_.add_argument("--backbone", type=str, choices=BackboneRegistry.get_all_names(), default="ncsnpp")
          parser_.add_argument("--ode", type=str, choices=ODERegistry.get_all_names(), default="flowmatching")    
          parser_.add_argument("--no_wandb", action='store_true', help="Turn off logging to W&B, using local default logger instead")
          
          
     temp_args, _ = base_parser.parse_known_args()

     # Add specific args for VFModel, pl.Trainer, the SDE class and backbone DNN class
     backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
     ode_class = ODERegistry.get_by_name(temp_args.ode)
     parser = pl.Trainer.add_argparse_args(parser)
     VFModel.add_argparse_args(
          parser.add_argument_group("VFModel", description=VFModel.__name__))
     ode_class.add_argparse_args(
          parser.add_argument_group("ODE", description=ode_class.__name__))
     backbone_cls.add_argparse_args(
          parser.add_argument_group("Backbone", description=backbone_cls.__name__))
     # Add data module args
     data_module_cls = SpecsDataModule
     data_module_cls.add_argparse_args(
          parser.add_argument_group("DataModule", description=data_module_cls.__name__))
     # Parse args and separate into groups
     args = parser.parse_args()
     arg_groups = get_argparse_groups(parser)
     dataset = os.path.basename(os.path.normpath(args.base_dir))
     # Initialize logger, trainer, model, datamodule
     model = VFModel(
          backbone=args.backbone, ode=args.ode, data_module_cls=data_module_cls,
          **{
               **vars(arg_groups['VFModel']),
               **vars(arg_groups['ODE']),
               **vars(arg_groups['Backbone']),
               **vars(arg_groups['DataModule'])
          }
     )
     # Set up logger configuration
     if args.no_wandb:
          logger = TensorBoardLogger(save_dir="logs", name="tensorboard")
     else:
          if ode_class.__name__ == "FLOWMATCHING":
               name_save_dir_path = f"mode_{args.mode_}_dataset_{dataset}_sigma_min_{args.sigma_min}_sigma_max_{args.sigma_max}_T_rev_{args.T_rev}_t_eps_{args.t_eps}"
               logger = WandbLogger(project=f"FLOWSE_CONDITION_EXPLORING", log_model=True, save_dir="logs", name=name_save_dir_path)
          
          
          else:
               raise ValueError(f"{ode_class.__name__}에 대한 configuration이 만들어지지 않았음")
          logger.experiment.log_code(".")

     # Set up callbacks for logger

     model_dirpath = f"logs/{name_save_dir_path}_{logger.version}"

     checkpoint_callback_last = ModelCheckpoint(dirpath=model_dirpath,
          save_last=True, filename='{epoch}-last')
     checkpoint_callback_pesq = ModelCheckpoint(dirpath=model_dirpath,          save_top_k=10, monitor="pesq", mode="max", filename='{epoch}-{pesq:.2f}')
     checkpoint_callback_valid_loss = ModelCheckpoint(dirpath=model_dirpath,  save_top_k=10, monitor="valid_loss", mode="min", filename='{epoch}-{valid_loss:.2f}')
     checkpoint_callback_si_sdr = ModelCheckpoint(dirpath=model_dirpath, save_top_k=10, monitor="si_sdr", mode="max", filename='{epoch}-{si_sdr:.2f}')
     checkpoint_callback_estoi = ModelCheckpoint(dirpath=model_dirpath,          save_top_k=10, monitor="estoi", mode="max", filename='{epoch}-{estoi:.2f}')

     early_stopping_callback = EarlyStopping(monitor="valid_loss",patience=1000, mode="min",verbose=True)
     callbacks = [checkpoint_callback_estoi, checkpoint_callback_valid_loss, checkpoint_callback_pesq, early_stopping_callback, checkpoint_callback_last, checkpoint_callback_si_sdr]

     # Initialize the Trainer and the DataModule
     trainer = pl.Trainer.from_argparse_args(
          arg_groups['pl.Trainer'],
          accelerator='gpu', strategy=DDPPlugin(find_unused_parameters=False), gpus=[0,1], auto_select_gpus=False, 
          logger=logger, log_every_n_steps=10, num_sanity_val_steps=1, max_epochs=1000,
          callbacks=callbacks
     )

     # Train model
     trainer.fit(model)

   