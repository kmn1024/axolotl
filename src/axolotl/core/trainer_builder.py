"""
Builder for the training args and trainer
"""

import abc
import importlib
import logging
import math
import os
import sys
import wandb
import numpy as np
from abc import abstractmethod
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional, Union

import torch
import transformers
from datasets import Dataset
from datasets.distributed import split_dataset_by_node
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler, OneCycleLR
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
from transformers.trainer_pt_utils import SequentialDistributedSampler
from transformers.trainer_utils import has_length, seed_worker

from axolotl.monkeypatch.relora import ReLoRACallback, ReLoRAScheduler
from axolotl.utils.callbacks import (
    EvalFirstStepCallback,
    GPUStatsCallback,
    SaveAxolotlConfigtoWandBCallback,
    SaveBetterTransformerModelCallback,
    bench_eval_callback_factory,
    log_prediction_callback_factory,
)
from axolotl.utils.collators import DataCollatorForSeq2Seq
from axolotl.utils.dataloader import MultipackDistributedDataloader, StreamingMultipackDistributedDataloaderNew
from axolotl.utils.schedulers import get_cosine_schedule_with_quadratic_warmup
from axolotl.utils.distributed import (
    is_main_process,
)


try:
    import torch._dynamo  # pylint: disable=ungrouped-imports
except ImportError:
    pass

LOG = logging.getLogger("axolotl.core.trainer_builder")


class CandidatePenaltyCrossEntropyCriterion():
    """Applies a (1-p(x_nt)) loss to each negative target ('candidate') x_nt."""

    def __init__(self, tokenizer):
        self.padding_idx = tokenizer.pad_token_id
        self.IGNORE_TOKEN_ID = -100  # Copied from prompt_strategies

    def forward(self, batched_inputs, batched_pred_logits):
        batched_labels = batched_inputs['labels']
        B, seq_len = batched_labels.shape
        B2, seq_len2, vocab_size2 = batched_pred_logits.shape
        assert B == B2 and seq_len == seq_len2, f'{batched_labels.shape}, {batched_labels.shape}'
        unliklihood_losses = torch.tensor([0] * B, dtype=torch.float)
        for i in range(B):
            target = batched_labels[i]
            shift_targets = target[..., 1:].contiguous()
            shift_targets = shift_targets.masked_fill(shift_targets == self.IGNORE_TOKEN_ID, self.padding_idx)
            mask_ignore = shift_targets != self.padding_idx
            shift_targets = shift_targets[mask_ignore]

            shift_logits = batched_pred_logits[i][..., :-1, :].contiguous()
            shift_lprobs = F.log_softmax(shift_logits, dim=-1)
            shift_lprobs = shift_lprobs.view(-1, shift_lprobs.size(-1))
            shift_lprobs = shift_lprobs[mask_ignore]

            # # Sanity check that this is the same as primary_loss.
            # sanity_mle_loss = F.nll_loss(
            #     shift_lprobs,
            #     shift_targets,
            #     reduction='mean')

            # -- unliklihood loss
            # Maximize (1 - p(x_nt)) for negative target tokens x_nt (equivalently minimize -log(1-p(x_nt)))

            # - form negative targets
            with torch.no_grad():
                # E.g. DABCC | D | EFFGD => {A,B,C} are negative targets.
                # Make 'the triangle'.
                ctx_cands = shift_targets.unsqueeze(0).repeat(shift_targets.size(0), 1)
                rows, cols = torch.triu_indices(shift_targets.size(0), shift_targets.size(0))
                ctx_cands[rows, cols] = self.padding_idx
                # Don't include the target for that timestep as a negative target.
                ctx_cands = ctx_cands.masked_fill(ctx_cands == shift_targets.unsqueeze(1), self.padding_idx)
                negative_targets = torch.zeros_like(shift_lprobs).scatter_(1, ctx_cands, 1)

            # - compute loss
            one_minus_probs = torch.clamp((1.0 - shift_lprobs.exp()), min=1e-5)
            unliklihood_loss = -torch.log(one_minus_probs)*negative_targets
            unliklihood_losses[i] = unliklihood_loss.sum(1).mean()
        return unliklihood_losses.mean()


@dataclass
class AxolotlTrainingArguments(TrainingArguments):
    """
    Extend the base TrainingArguments for axolotl helpers
    """

    local_streaming_datasets: bool = field(
        default=False,
        metadata={"help": "Whether datasets are streaming, from local files."},
    )
    lr_quadratic_warmup: bool = field(
        default=False,
        metadata={"help": "Use quadratic warmup for cosine scheduling."},
    )
    sample_packing: bool = field(
        default=False,
        metadata={"help": "Use sample packing for efficient training."},
    )
    eval_sample_packing: Optional[bool] = field(
        default=None,
        metadata={"help": "Use sample packing for efficient evals."},
    )
    sample_packing_efficiency: float = field(
        default=1.0,
        metadata={"help": "Sample packing efficiency for calculating batch length."},
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "The maximum sequence length the model can handle"},
    )
    sample_packing_seq_len_multiplier: int = field(
        default=1,
        metadata={"help": "the multiplier for the max len for packed sequences"},
    )
    relora_steps: Optional[int] = field(
        default=None,
        metadata={"help": "how often to reset for ReLoRA"},
    )
    relora_warmup_steps: Optional[int] = field(
        default=None,
        metadata={"help": "how many warmup steps to take after reset for ReLoRA"},
    )
    bench_split: Optional[str] = field(
        default="eval", metadata={"help": "The benchmark split to run on"}
    )
    bench_dataset: Optional[str] = field(
        default="pharaouk/dharma-1/dharma_1_mini.json",
        metadata={
            "help": "Benchmark dataset to use: options are `mmlu-zs`, `mmlu-fs`, or the full path to the dataset file"
        },
    )
    do_bench_eval: Optional[bool] = field(
        default=False, metadata={"help": "Whether to run the Benchmark evaluation."}
    )
    max_bench_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, only evaluates on `max_bench_samples` of the benchmark dataset."
        },
    )
    bench_source_max_len: int = field(
        default=2048, metadata={"help": "Maximum source sequence length for bench."}
    )


class AxolotlTrainer(Trainer):
    """
    Extend the base Trainer for axolotl helpers
    """

    args = None  # type: AxolotlTrainingArguments

    def __init__(self, *args, bench_data_collator=None, **kwargs):
        self.bench_data_collator = bench_data_collator
        self.primary_loss_tmp, self.secondary_loss_tmp = [], []
        self.is_accumulating_eval = False
        super().__init__(*args, **kwargs)

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer = None
    ):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
            optimizer (torch.optim.Optimizer): The training optimizer
        """

        # fmt: off
        if self.lr_scheduler is None:  # type: ignore  # pylint: disable=access-member-before-definition
            # fmt: on
            if (
                self.args.lr_scheduler_type == "cosine"
                and self.args.lr_quadratic_warmup is True
            ):
                self.lr_scheduler = get_cosine_schedule_with_quadratic_warmup(  # pylint: disable=attribute-defined-outside-init
                    optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                )
            else:
                self.lr_scheduler = super().create_scheduler(num_training_steps, optimizer)
        return self.lr_scheduler

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.args.world_size > 1 and self.args.sample_packing:
            return DistributedSampler(
                self.train_dataset,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                seed=self.args.seed,
            )
        return super()._get_train_sampler()

    def _get_eval_sampler(
        self, eval_dataset: Dataset
    ) -> Optional[torch.utils.data.Sampler]:
        if self.args.world_size > 1:
            return SequentialDistributedSampler(
                eval_dataset,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                batch_size=self.args.per_device_eval_batch_size,
            )
        return super()._get_eval_sampler(eval_dataset)

    def get_train_dataloader(self) -> Union[DataLoader, MultipackDistributedDataloader, StreamingMultipackDistributedDataloaderNew]:
        if self.args.sample_packing and not self.args.local_streaming_datasets:
            train_sampler = self._get_train_sampler()
            return self.accelerator.prepare(
                MultipackDistributedDataloader(
                    self.train_dataset,
                    batch_size=self._train_batch_size,
                    seq_max_length=self.args.max_seq_length,
                    collate_fn=self.data_collator,
                    sampler=train_sampler,
                    packing_efficiency_estimate=self.args.sample_packing_efficiency,
                    sample_packing_seq_len_multiplier=self.args.sample_packing_seq_len_multiplier,
                    device_count=int(os.environ.get("WORLD_SIZE", 1)),
                )
            )
        elif self.args.sample_packing and self.args.local_streaming_datasets:
            print(f'get_train_dataloader input shards: {self.train_dataset.n_shards}')
            self.train_dataset = split_dataset_by_node(self.train_dataset,
                                                       rank=self.args.process_index,
                                                       world_size=self.args.world_size)
            print(f'split_dataset_by_node shards: {self.train_dataset.n_shards}')
            prefetch_factor = 3
            return self.accelerator.prepare(
                StreamingMultipackDistributedDataloaderNew(
                    self.train_dataset, self.data_collator,
                    self.args.dataloader_num_workers, prefetch_factor,
                    self.args.max_seq_length, self._train_batch_size,
                    self.args.sample_packing_seq_len_multiplier
                )
            )
        else:
            # Copied from transformers/trainer.py
            # return super().get_train_dataloader()
            train_dataset = self.train_dataset
            data_collator = self.data_collator
            if isinstance(train_dataset, Dataset):
                train_dataset = self._remove_unused_columns(train_dataset, description="training")
            else:
                data_collator = self._get_collator_with_removed_columns(data_collator, description="training")
            dataloader_params = {
                "batch_size": self._train_batch_size,
                "collate_fn": data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "prefetch_factor": 3,
            }
            if not isinstance(train_dataset, torch.utils.data.IterableDataset):
                dataloader_params["sampler"] = self._get_train_sampler()
                dataloader_params["drop_last"] = self.args.dataloader_drop_last
                dataloader_params["worker_init_fn"] = seed_worker
            return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))


    def get_eval_dataloader(
        self, eval_dataset: Optional[Dataset] = None
    ) -> Union[DataLoader, MultipackDistributedDataloader]:
        if self.args.sample_packing and self.args.eval_sample_packing is not False:
            eval_dataset = (
                eval_dataset if eval_dataset is not None else self.eval_dataset
            )

            eval_sampler = self._get_eval_sampler(eval_dataset)
            return self.accelerator.prepare(
                MultipackDistributedDataloader(
                    eval_dataset,
                    batch_size=self.args.eval_batch_size,
                    seq_max_length=self.args.max_seq_length,
                    collate_fn=self.data_collator,
                    sampler=eval_sampler,
                    packing_efficiency_estimate=self.args.sample_packing_efficiency,
                    sample_packing_seq_len_multiplier=self.args.eval_batch_size,
                    device_count=int(os.environ.get("WORLD_SIZE", 1)),
                )
            )
        return super().get_eval_dataloader(eval_dataset)

    def _get_bench_sampler(
        self, bench_dataset: Dataset
    ) -> Optional[torch.utils.data.Sampler]:
        if self.args.world_size <= 1:
            return SequentialSampler(bench_dataset)
        return None

    def get_bench_dataloader(
        self,
        bench_dataset: Dataset,
    ) -> Union[DataLoader, MultipackDistributedDataloader]:
        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": self.bench_data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(bench_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_bench_sampler(bench_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        return DataLoader(bench_dataset, **dataloader_params)
        # return self.accelerator.prepare(DataLoader(bench_dataset, **dataloader_params))

    def compute_loss(self, model, inputs, return_outputs=False):
        primary_loss, model_outputs = super().compute_loss(model, inputs, return_outputs=True)
        candidate_penalty = CandidatePenaltyCrossEntropyCriterion(self.tokenizer)
        secondary_loss = candidate_penalty.forward(inputs, model_outputs['logits'])
        loss = primary_loss + 1.0 * secondary_loss
        if is_main_process():
            if model.training:
                if self.is_accumulating_eval:
                    # Just finished eval loop.
                    assert self.primary_loss_tmp, 'Should not be empty!'
                    step_primary_loss = np.mean(self.primary_loss_tmp)
                    step_secondary_loss = np.mean(self.secondary_loss_tmp)
                    LOG.info(f'eval_primary_loss:{step_primary_loss}, eval_secondary_loss:{step_secondary_loss}')
                    wandb.log({"eval/primary_loss": step_primary_loss, "eval/secondary_loss": step_secondary_loss})
                    self.primary_loss_tmp = []
                    self.secondary_loss_tmp = []
                    self.is_accumulating_eval = False
                self.primary_loss_tmp.append(primary_loss.data.item())
                self.secondary_loss_tmp.append(secondary_loss.data.item())
                if self.primary_loss_tmp and len(self.primary_loss_tmp) % (self.args.gradient_accumulation_steps * self.args.logging_steps) == 0:
                    step_primary_loss = np.mean(self.primary_loss_tmp)
                    step_secondary_loss = np.mean(self.secondary_loss_tmp)
                    LOG.info(f'train_primary_loss:{step_primary_loss}, train_secondary_loss:{step_secondary_loss}')
                    wandb.log({"train/primary_loss": step_primary_loss, "train/secondary_loss": step_secondary_loss})
                    self.primary_loss_tmp = []
                    self.secondary_loss_tmp = []
            else:
                if not self.is_accumulating_eval:
                    # Just finished training loop. We may miss a small number of training logs here, who cares.
                    self.primary_loss_tmp = []
                    self.secondary_loss_tmp = []
                    self.is_accumulating_eval = True
                self.primary_loss_tmp.append(primary_loss.data.item())
                self.secondary_loss_tmp.append(secondary_loss.data.item())
        return (loss, model_outputs) if return_outputs else loss
        #return super().compute_loss(model, inputs, return_outputs=return_outputs)



class OneCycleLRSchedulerTrainer(AxolotlTrainer):
    """
    Trainer subclass that uses the OneCycleLR scheduler
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_scheduler = None

    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        optimizer = self.optimizer if optimizer is None else optimizer
        num_warmup_steps = self.args.get_warmup_steps(num_training_steps)
        pct_start = num_warmup_steps / num_training_steps

        self.lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=self.args.learning_rate,
            total_steps=num_training_steps,
            pct_start=pct_start,
            div_factor=6,
        )

        return self.lr_scheduler


class ReLoRATrainer(AxolotlTrainer):
    """
    Trainer subclass that uses the OneCycleLR scheduler
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_scheduler = None

    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        optimizer = self.optimizer if optimizer is None else optimizer
        lr_scheduler = super().create_scheduler(num_training_steps, optimizer)

        if self.args.relora_steps:
            warmup_steps = (
                self.args.relora_warmup_steps if self.args.relora_warmup_steps else 10
            )
            self.lr_scheduler = ReLoRAScheduler(
                optimizer,
                lr_scheduler,
                self.args.relora_steps,
                warmup_steps,
            )
        else:
            self.lr_scheduler = lr_scheduler

        return self.lr_scheduler


class TrainerBuilderBase(abc.ABC):
    """
    Base class for trainer builder
    """

    _train_dataset = None
    _eval_dataset = None

    def __init__(self, cfg, model, tokenizer):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer

    @property
    def train_dataset(self):
        return self._train_dataset

    @train_dataset.setter
    def train_dataset(self, dataset):
        self._train_dataset = dataset

    @property
    def eval_dataset(self):
        return self._eval_dataset

    @eval_dataset.setter
    def eval_dataset(self, dataset):
        self._eval_dataset = dataset

    @abstractmethod
    def build(self, total_num_steps):
        pass

    @abstractmethod
    def get_callbacks(self):
        pass

    @abstractmethod
    def get_post_trainer_create_callbacks(self, trainer):
        """
        Callbacks added after the trainer is created, usually b/c these need access to the trainer
        """


class HFCausalTrainerBuilder(TrainerBuilderBase):
    """
    Build the HuggingFace training args/trainer for Causal models
    """

    def hook_pre_create_training_args(self, training_arguments_kwargs):
        # TODO
        return training_arguments_kwargs

    def hook_post_create_training_args(self, training_arguments):
        # TODO
        return training_arguments

    def hook_pre_create_trainer(self, trainer_kwargs, trainer_cls):
        # TODO
        return trainer_kwargs, trainer_cls

    def hook_post_create_trainer(self, trainer):
        # TODO
        return trainer

    def get_callbacks(self):
        callbacks = []
        callbacks.append(GPUStatsCallback(self.cfg))
        callbacks.append(EvalFirstStepCallback)

        if self.cfg.relora_steps:
            callbacks.append(ReLoRACallback(self.cfg))

        if (
            hasattr(self.model, "use_bettertransformer")
            and self.model.use_bettertransformer is True
        ):
            callbacks.append(SaveBetterTransformerModelCallback)

        if self.cfg.use_wandb:
            callbacks.append(
                SaveAxolotlConfigtoWandBCallback(self.cfg.axolotl_config_path)
            )

        return callbacks

    def get_post_trainer_create_callbacks(self, trainer):
        callbacks = []
        if self.cfg.use_wandb and self.cfg.eval_table_size > 0:
            LogPredictionCallback = log_prediction_callback_factory(
                trainer, self.tokenizer
            )
            callbacks.append(LogPredictionCallback(self.cfg))

        if self.cfg.do_bench_eval:
            callbacks.append(bench_eval_callback_factory(trainer, self.tokenizer))

        if self.cfg.early_stopping_patience:
            early_stop_cb = EarlyStoppingCallback(
                self.cfg.early_stopping_patience,
            )
            callbacks.append(early_stop_cb)

        return callbacks

    def _get_trainer_cls(self):
        if self.cfg.lr_scheduler == "one_cycle" and (
            self.cfg.fsdp or self.cfg.adapter == "qlora"
        ):
            return OneCycleLRSchedulerTrainer
        if self.cfg.relora_steps:
            return ReLoRATrainer
        return AxolotlTrainer

    def build(self, total_num_steps):
        warmup_steps = (
            self.cfg.warmup_steps
            if self.cfg.warmup_steps is not None
            else min(int(0.03 * total_num_steps), 100)
        )
        logging_steps = (
            self.cfg.logging_steps
            if self.cfg.logging_steps is not None
            else max(min(int(0.005 * total_num_steps), 10), 1)
        )

        training_arguments_kwargs = {}
        if self.cfg.bf16 == "full":
            training_arguments_kwargs["bf16_full_eval"] = True
        else:
            training_arguments_kwargs["bf16"] = self.cfg.bf16
        training_arguments_kwargs["fp16"] = (
            self.cfg.fp16 and not self.cfg.bf16
        ) or False
        training_arguments_kwargs["tf32"] = self.cfg.tf32
        training_arguments_kwargs["warmup_steps"] = warmup_steps
        training_arguments_kwargs["logging_steps"] = logging_steps

        if self.cfg.seed:
            training_arguments_kwargs["seed"] = self.cfg.seed

        if self.cfg.gradient_checkpointing:
            training_arguments_kwargs[
                "gradient_checkpointing"
            ] = self.cfg.gradient_checkpointing
        if self.cfg.fsdp:
            training_arguments_kwargs["fsdp"] = self.cfg.fsdp
            if self.cfg.fsdp_config:
                training_arguments_kwargs["fsdp_config"] = dict(self.cfg.fsdp_config)

        # deepspeed
        if self.cfg.deepspeed:
            training_arguments_kwargs["deepspeed"] = self.cfg.deepspeed
        if self.cfg.local_streaming_datasets is not None:
            training_arguments_kwargs[
                "local_streaming_datasets"
            ] = self.cfg.local_streaming_datasets
        if self.cfg.lr_quadratic_warmup is not None:
            training_arguments_kwargs[
                "lr_quadratic_warmup"
            ] = self.cfg.lr_quadratic_warmup

        if self.cfg.adam_beta1:
            training_arguments_kwargs["adam_beta1"] = self.cfg.adam_beta1
        if self.cfg.adam_beta2:
            training_arguments_kwargs["adam_beta2"] = self.cfg.adam_beta2
        if self.cfg.adam_epsilon:
            training_arguments_kwargs["adam_epsilon"] = self.cfg.adam_epsilon
        if self.cfg.max_grad_norm:
            training_arguments_kwargs["max_grad_norm"] = self.cfg.max_grad_norm

        if self.cfg.hub_model_id:
            training_arguments_kwargs["hub_model_id"] = self.cfg.hub_model_id
            training_arguments_kwargs["push_to_hub"] = True
            training_arguments_kwargs["hub_private_repo"] = True

            if self.cfg.hub_strategy:
                training_arguments_kwargs["hub_strategy"] = self.cfg.hub_strategy

        if self.cfg.save_safetensors:
            training_arguments_kwargs["save_safetensors"] = self.cfg.save_safetensors

        if self.cfg.sample_packing_eff_est:
            training_arguments_kwargs[
                "sample_packing_efficiency"
            ] = self.cfg.sample_packing_eff_est

        if self.cfg.eval_steps:
            training_arguments_kwargs["evaluation_strategy"] = "steps"
            training_arguments_kwargs["eval_steps"] = self.cfg.eval_steps
        elif self.cfg.evaluation_strategy:
            training_arguments_kwargs[
                "evaluation_strategy"
            ] = self.cfg.evaluation_strategy
        elif self.cfg.val_set_size == 0:
            # no eval set, so don't eval
            training_arguments_kwargs["evaluation_strategy"] = "no"
        else:
            # we have an eval set, but no steps defined, default to use epoch
            training_arguments_kwargs["evaluation_strategy"] = "epoch"

        if self.cfg.save_steps:
            training_arguments_kwargs["save_strategy"] = "steps"
            training_arguments_kwargs["save_steps"] = self.cfg.save_steps
        elif self.cfg.save_strategy:
            training_arguments_kwargs["save_strategy"] = self.cfg.save_strategy
        else:
            # default to saving each epoch if not defined
            training_arguments_kwargs["save_strategy"] = "epoch"

        if self.cfg.do_bench_eval:
            training_arguments_kwargs["do_bench_eval"] = self.cfg.do_bench_eval
            if self.cfg.bench_dataset:
                training_arguments_kwargs["bench_dataset"] = self.cfg.bench_dataset
        if self.cfg.metric_for_best_model:
            training_arguments_kwargs[
                "metric_for_best_model"
            ] = self.cfg.metric_for_best_model
        if self.cfg.greater_is_better:
            training_arguments_kwargs["greater_is_better"] = self.cfg.greater_is_better

        if self.cfg.torch_compile:
            if torch.__version__ < "2.1.0":  # pylint: disable=protected-access
                LOG.warning("torch>=2.1.0 required for torch_compile to work properly")
            elif torch._dynamo:  # pylint: disable=protected-access
                torch._dynamo.config.suppress_errors = (  # pylint: disable=protected-access
                    True
                )
                training_arguments_kwargs["torch_compile"] = self.cfg.torch_compile
                if self.cfg.torch_compile_backend:
                    training_arguments_kwargs[
                        "torch_compile_backend"
                    ] = self.cfg.torch_compile_backend

        # DDP Config
        if self.cfg.ddp_timeout:
            training_arguments_kwargs["ddp_timeout"] = self.cfg.ddp_timeout
        # see https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
        if self.cfg.ddp_bucket_cap_mb:
            training_arguments_kwargs["ddp_bucket_cap_mb"] = self.cfg.ddp_bucket_cap_mb
        if self.cfg.ddp_broadcast_buffers is not None:
            training_arguments_kwargs[
                "ddp_broadcast_buffers"
            ] = self.cfg.ddp_broadcast_buffers

        # these are all the "standard" kwargs that are def used
        training_arguments_kwargs["max_steps"] = (
            total_num_steps if self.cfg.max_steps else -1
        )
        training_arguments_kwargs["max_seq_length"] = self.cfg.sequence_len
        training_arguments_kwargs[
            "per_device_train_batch_size"
        ] = self.cfg.micro_batch_size
        training_arguments_kwargs[
            "per_device_eval_batch_size"
        ] = self.cfg.eval_batch_size
        training_arguments_kwargs[
            "gradient_accumulation_steps"
        ] = self.cfg.gradient_accumulation_steps
        training_arguments_kwargs[
            "eval_accumulation_steps"
        ] = self.cfg.gradient_accumulation_steps
        training_arguments_kwargs["num_train_epochs"] = self.cfg.num_epochs
        training_arguments_kwargs["learning_rate"] = self.cfg.learning_rate
        training_arguments_kwargs["output_dir"] = self.cfg.output_dir
        training_arguments_kwargs["save_total_limit"] = (
            self.cfg.save_total_limit if self.cfg.save_total_limit else 4
        )
        training_arguments_kwargs["load_best_model_at_end"] = (
            (
                self.cfg.load_best_model_at_end is not False
                or self.cfg.early_stopping_patience
            )
            and self.cfg.val_set_size > 0
            and self.cfg.save_steps
            and self.cfg.eval_steps
            and self.cfg.save_steps % self.cfg.eval_steps == 0
        ) or False
        training_arguments_kwargs["ddp_find_unused_parameters"] = (
            False if self.cfg.ddp else None
        )
        training_arguments_kwargs["group_by_length"] = self.cfg.group_by_length
        training_arguments_kwargs["report_to"] = "wandb" if self.cfg.use_wandb else None
        training_arguments_kwargs["run_name"] = (
            self.cfg.wandb_run_id if self.cfg.use_wandb else None
        )
        training_arguments_kwargs["optim"] = (
            self.cfg.optimizer if self.cfg.optimizer else "adamw_hf"
        )
        training_arguments_kwargs["lr_scheduler_type"] = (
            self.cfg.lr_scheduler
            if self.cfg.lr_scheduler
            and self.cfg.lr_scheduler not in ("one_cycle", "log_sweep")
            else "cosine"
        )
        training_arguments_kwargs["weight_decay"] = (
            self.cfg.weight_decay if self.cfg.weight_decay is not None else 0.0
        )
        training_arguments_kwargs["sample_packing"] = (
            self.cfg.sample_packing if self.cfg.sample_packing else False
        )
        training_arguments_kwargs["eval_sample_packing"] = (
            self.cfg.sample_packing if self.cfg.sample_packing else False
        )
        training_arguments_kwargs[
            "sample_packing_seq_len_multiplier"
        ] = self.cfg.micro_batch_size
        training_arguments_kwargs["relora_steps"] = self.cfg.relora_steps
        training_arguments_kwargs["relora_warmup_steps"] = self.cfg.relora_warmup_steps
        if self.cfg.dataloader_num_workers:
            training_arguments_kwargs["dataloader_num_workers"] = self.cfg.dataloader_num_workers
        training_arguments_kwargs = self.hook_pre_create_training_args(
            training_arguments_kwargs
        )
        training_args = (
            AxolotlTrainingArguments(  # pylint: disable=unexpected-keyword-arg
                **training_arguments_kwargs,
            )
        )
        training_args.dispatch_batches = False
        training_args = self.hook_post_create_training_args(training_args)
        trainer_kwargs = {}

        if self.cfg.optimizer == "adamw_anyprecision":
            if Path(self.cfg.torchdistx_path).exists():
                sys.path.append(self.cfg.torchdistx_path)
                importlib.import_module("torchdistx")

        data_collator_kwargs = {
            "padding": True,  # True/"longest" is the default
        }
        if self.cfg.pad_to_sequence_len:
            data_collator_kwargs["pad_to_multiple_of"] = 64 * math.ceil(
                self.cfg.sequence_len / 64
            )
        else:
            # A100 is best at 64, while others at 8. Let's use the larger so we don't have to check
            # https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
            data_collator_kwargs["pad_to_multiple_of"] = 64

        if self.cfg.is_llama_derived_model and self.cfg.landmark_attention:
            from axolotl.monkeypatch.llama_landmark_attn import (
                add_mem_tokens,
                get_mem_id,
                set_model_mem_id,
            )

            set_model_mem_id(self.model, self.tokenizer)

            LOG.info("Adding landmark attention tokens to dataset")

            for dataset in [self.train_dataset, self.eval_dataset]:
                dataset = dataset.map(
                    partial(
                        add_mem_tokens, mem_freq=50, mem_id=get_mem_id(self.tokenizer)
                    ),
                    batched=False,
                    num_proc=32,
                )

        trainer_cls = self._get_trainer_cls()
        trainer_kwargs, trainer_cls = self.hook_pre_create_trainer(
            trainer_kwargs, trainer_cls
        )
        trainer = trainer_cls(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            args=training_args,
            data_collator=DataCollatorForSeq2Seq(
                self.tokenizer,
                return_tensors="pt",
                **data_collator_kwargs,
            ),
            bench_data_collator=transformers.DataCollatorForSeq2Seq(
                self.tokenizer,
                return_tensors="pt",
                **data_collator_kwargs,
            ),
            callbacks=self.get_callbacks(),
            tokenizer=self.tokenizer,
            **trainer_kwargs,
        )
        trainer = self.hook_post_create_trainer(trainer)
        for callback in self.get_post_trainer_create_callbacks(trainer):
            trainer.add_callback(callback)

        return trainer
