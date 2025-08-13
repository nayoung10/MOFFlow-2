import os
import random
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# Pytorch lightning imports
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning_fabric.utilities.seed import seed_everything
from data.dataset import MOFDataset
from data.dataloader import MOFDatamodule
from models.flow_module import FlowModule
from utils.environment import PROJECT_ROOT
from utils import experiment as eu
from utils.callbacks import TensorBoardCallback, LMDBCleanupCallback
from utils.ema import EMACallback, EMAModelCheckpoint

log = eu.get_pylogger(__name__)
torch.set_float32_matmul_precision('high')
OmegaConf.register_new_resolver("eval", lambda expr: eval(expr))


class Experiment:

    def __init__(self, *, cfg: DictConfig):
        self._cfg = cfg
        self._data_cfg = cfg.data
        self._exp_cfg = cfg.experiment
        self._setup_dataset()
        self._datamodule: LightningDataModule = MOFDatamodule(
            data_cfg=self._data_cfg,
            train_dataset=self._train_dataset,
            valid_dataset=self._valid_dataset
        )
        log.info(f"Training with {self._exp_cfg.num_devices} devices.")
        self._module: LightningModule = FlowModule(self._cfg)

        if self._exp_cfg.seed is not None:
            log.info(f'Setting seed to {self._exp_cfg.seed}')
            seed_everything(self._exp_cfg.seed, workers=True)

    def _setup_dataset(self):
        self._train_dataset = MOFDataset(
            dataset_cfg=self._data_cfg,
            split='train',
        )
        self._valid_dataset = MOFDataset(
            dataset_cfg=self._data_cfg,
            split='val'
        )
        
    def train(self):
        callbacks = []
        if self._exp_cfg.debug:
            log.info("Debug mode.")
            logger = None
            self._exp_cfg.num_devices = 1
            self._data_cfg.loader.num_workers = 0
        else:
            # Register callbacks
            if self._exp_cfg.use_ema:
                callbacks.append(EMACallback(**self._exp_cfg.ema))
                callbacks.append(EMAModelCheckpoint(**self._exp_cfg.checkpointer))
            else:
                callbacks.append(ModelCheckpoint(**self._exp_cfg.checkpointer))
            callbacks.append(LearningRateMonitor(logging_interval='step'))
            callbacks.append(LMDBCleanupCallback(self._train_dataset))
            callbacks.append(LMDBCleanupCallback(self._valid_dataset))

            if self._cfg.logger == 'wandb':
                logger = WandbLogger(
                    **self._cfg.wandb,
                )
                logger.watch(
                    self._module,
                    log=self._cfg.wandb_watch.log,
                    log_freq=self._cfg.wandb_watch.log_freq
                )
            elif self._cfg.logger == 'tensorboard':
                logger = TensorBoardLogger(
                    **self._cfg.tensorboard
                )
                tb_callback = TensorBoardCallback(
                    **self._cfg.callbacks.tensorboard
                )
                callbacks.append(tb_callback)
            
            # Checkpoint directory.
            ckpt_dir = self._exp_cfg.checkpointer.dirpath
            os.makedirs(ckpt_dir, exist_ok=True)
            log.info(f"Checkpoints saved to {ckpt_dir}")

            # Save config only for main process.
            local_rank = os.environ.get('LOCAL_RANK', 0)
            if local_rank == 0:
                # Save config
                cfg_path = os.path.join(ckpt_dir, 'config.yaml')
                with open(cfg_path, 'w') as f:
                    OmegaConf.save(config=self._cfg, f=f.name)

                # Flatten config
                cfg_dict = OmegaConf.to_container(self._cfg, resolve=True)
                flat_cfg = dict(eu.flatten_dict(cfg_dict))

                # Log config
                logger.log_hyperparams(flat_cfg)
        
        is_dynamic_sampler = self._data_cfg.loader.sampler == 'dynamic'
        use_distributed_sampler = not is_dynamic_sampler
        trainer = Trainer(
            **self._exp_cfg.trainer,
            callbacks=callbacks,
            logger=logger,
            use_distributed_sampler=use_distributed_sampler,
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=self._exp_cfg.num_devices,
        )
        trainer.fit(
            model=self._module,
            datamodule=self._datamodule,
            ckpt_path=self._exp_cfg.warm_start
        )


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="base.yaml")
def main(cfg: DictConfig):

    if cfg.experiment.warm_start is not None and cfg.experiment.warm_start_cfg_override:
        # Loads warm start config.
        warm_start_cfg_path = os.path.join(
            os.path.dirname(cfg.experiment.warm_start), 'config.yaml')
        warm_start_cfg = OmegaConf.load(warm_start_cfg_path)

        # Warm start config may not have latest fields in the base config.
        # Add these fields to the warm start config.
        OmegaConf.set_struct(cfg.model, False)
        OmegaConf.set_struct(warm_start_cfg.model, False)
        cfg.model = OmegaConf.merge(cfg.model, warm_start_cfg.model)
        OmegaConf.set_struct(cfg.model, True)
        log.info(f'Loaded warm start config from {warm_start_cfg_path}')

    exp = Experiment(cfg=cfg)
    exp.train()

if __name__ == "__main__":
    main()
