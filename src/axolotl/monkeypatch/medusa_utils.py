from transformers import (
    PretrainedConfig,
    TrainerCallback,
)
import logging
import warnings
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from axolotl.utils.dict import DictDefault
from axolotl.utils.distributed import is_main_process
import axolotl
from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index
import types
import math
import wandb

logger = LOG = logging.getLogger("axolotl.monkeypatch.medusa")

class MedusaConfig(PretrainedConfig):
    """
    Configuration class for Medusa model.
    Args:
        medusa_num_heads (int, optional): Number of heads for the Medusa layer. Default is 2.
        medusa_num_layers (int, optional): Number of Medusa layers. Default is 1.
        base_model_name_or_path (str, optional): The name or path of the base model. Default is "lmsys/vicuna-7b-v1.3".
        num_unfreezed_layers (int, optional): Number of layers to unfreeze. Default is 0.
        **kwargs: Additional keyword arguments to be passed to the parent class constructor.
    """

    def __init__(
        self,
        medusa_num_heads=4,
        medusa_num_layers=1,
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path


class ResBlock(nn.Module):
    """
    A Residual Block module.
    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.
    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))

def add_medusa_heads(
    self,
    medusa_num_heads=4,
    medusa_num_layers=0,
):
    """
    Args:
        self (nn.Module): The base language model to be used.
        medusa_num_heads (int, optional): Number of additional tokens to predict. Defaults to 3.
        medusa_num_layers (int, optional): Number of ResBlock layers for each Medusa head. Defaults to 0.
    """
    hidden_size = self.lm_head.weight.shape[-1]
    vocab_size = self.lm_head.weight.shape[0]
    LOG.info(f'Medusa detected hidden_size: {hidden_size}, vocab_size: {vocab_size}')
    self.medusa_num_heads = medusa_num_heads
    # Create a list of Medusa heads
    self.medusa_head = nn.ModuleList(
        [
            nn.Sequential(
                *([ResBlock(hidden_size)] * medusa_num_layers),
                nn.Linear(hidden_size, vocab_size, bias=False),
            )
            for _ in range(medusa_num_heads)
        ]
    )

    # Ensure medusa_head's dtype and device align with the base_model
    self.medusa_head.to(self.dtype).to(self.device)

    for i in range(medusa_num_heads):
        # Initialize the weights of each medusa_head using the base model's weights
        self.medusa_head[i][-1].weight.data[:] = self.lm_head.weight.data[:]

    self.old_forward = self.forward

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        medusa_return: bool = False,
    ):
        """Forward pass of the MedusaModel.
        Returns:
            torch.Tensor: A tensor containing predictions from all Medusa heads.
            (Optional) Original predictions from the base model's LM head.
        """
        # LOG.debug("medusa_return: %s", medusa_return)
        if not medusa_return:
            return self.old_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # Pass input through the base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0].clone()
        medusa_logits = [self.lm_head(hidden_states)]
        for i in range(self.medusa_num_heads):
            medusa_logits.append(self.medusa_head[i](hidden_states))
        return torch.stack(medusa_logits, dim=0)

    self.forward = types.MethodType(forward, self)

def replace_compute_loss(
    medusa_decay_coefficient, 
    medusa_logging=False,
):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the training loss for the model.
        Args:
            model (torch.nn.Module): The model for which to compute the loss.
            inputs (dict): The input data, including input IDs, attention mask, and labels.
            return_outputs (bool): Whether to return model outputs along with the loss.
        Returns:
            Union[float, Tuple[float, torch.Tensor]]: The computed loss, optionally with model outputs.
        """

        logits = model(
            **inputs,
            medusa_return=True,
        )
        labels = inputs["labels"]
        # Shift so that tokens < n predict n
        loss = 0
        loss_fct = CrossEntropyLoss()
        log = {}
        medusa = logits.shape[0]
        for i in range(medusa):
            medusa_logits = logits[i, :, : -(1 + i)].contiguous()
            medusa_labels = labels[..., 1 + i :].contiguous()
            medusa_logits = medusa_logits.view(-1, logits.shape[-1])
            medusa_labels = medusa_labels.view(-1)
            medusa_labels = medusa_labels.to(medusa_logits.device)
            loss_i = loss_fct(medusa_logits, medusa_labels)
            # Add decay coefficient to the loss
            if i == 0:
                # The original LM head.
                pass
            else:
                loss += loss_i * medusa_decay_coefficient ** i
            not_ignore = medusa_labels.ne(IGNORE_TOKEN_ID)
            medusa_labels = medusa_labels[not_ignore]

            # Add top-k accuracy
            for k in range(1, 10):
                _, topk = medusa_logits.topk(k, dim=-1)
                topk = topk[not_ignore]
                correct = topk.eq(medusa_labels.unsqueeze(-1)).any(-1)
                log[f"medusa{i}_top{k}"] = correct.float().mean().item()

            log[f"medusa{i}_loss"] = loss_i.item()
        # self.log(log)
        # Add prefix to the log
        if model.training:
            prefix = "train"
        else:
            prefix = "eval"
        log = {f"{prefix}/{k}": v for k, v in log.items()}
        if medusa_logging and self.state.is_world_process_zero:
            # Hardcoded for now
            wandb.log({
                **log,
                "train/global_step": self.state.global_step,
            })
        return (loss, logits) if return_outputs else loss
    axolotl.core.trainer_builder.AxolotlTrainer.compute_loss = compute_loss


def replace_create_optimizer(
    medusa_lr_multiplier,
):
    # Copy and simplified from transformers.Trainer.create_optimizer
    from transformers.trainer import Trainer
    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model
        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            # Separately set lr for medusa_head
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and "medusa_head" not in n)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and "medusa_head" in n)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate * medusa_lr_multiplier,
                },

                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer
    axolotl.core.trainer_builder.AxolotlTrainer.create_optimizer = create_optimizer