"""Module containing the PygmalionPromptTokenizingStrategy and PygmalionPrompter class"""

import copy
import logging
from collections import defaultdict
from typing import Generator, List, Tuple

from axolotl.prompt_tokenizers import (
    PromptTokenizingStrategy,
    parse_tokenized_to_result,
    tokenize_prompt_default,
)

LOG = logging.getLogger("axolotl")

IGNORE_TOKEN_ID = -100


class PygmalionPromptTokenizingStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for Pygmalion.
    """

    bot_prefix_token_ids: List[int] = []

    def __init__(self, prompter, tokenizer, *args, **kwargs):
        super().__init__(prompter, tokenizer, *args, **kwargs)
        res = self._tokenize("<|model|>", add_eos_token=False, strip_bos_token=True)
        self.bot_prefix_token_ids = res["input_ids"]

    def tokenize_prompt(self, prompt):
        result, current_len = tokenize_prompt_default()
        conversations = self.prompter.build_prompt(prompt["conversations"])
    
        started_dialogue = False
        while (next_item := next(conversations, None)) is not None:
            role, message = next_item
            if role == "system":
                prefix = "<|system|>"
                # this should include a bos token, no eos token, strip trailing "\n<START>"
                if message.endswith("\n<START>"):
                    message = message[:-8]
                res = self._tokenize(
                    prefix + "Persona: " + message.strip(),
                    add_eos_token=False,
                    strip_bos_token=False,
                )
                # everything from this is masked out from the labels
                labels = [IGNORE_TOKEN_ID] * len(res["input_ids"])
                # pylint: disable=duplicate-code
                result, current_len = parse_tokenized_to_result(
                    result,
                    current_len,
                    res,
                    labels,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            elif role == "human":
                next_next_item = next(conversations, None)
                assert next_next_item is not None
                bot_role, bot_message = next_next_item
                assert bot_role == "bot"
               
                user_prefix = "<|user|>"
                user_res = self._tokenize(
                    user_prefix + " " + message.strip(),
                    add_eos_token=False,
                    strip_bos_token=True,
                )
                # everything from this is masked out from the labels
                user_labels = [IGNORE_TOKEN_ID] * len(user_res["input_ids"])

                bot_prefix = "<|model|>"
                bot_res = self._tokenize(
                    bot_prefix + " " + bot_message.strip(),
                    add_eos_token=True,
                    strip_bos_token=True,
                )
                # mask out the prefix token, rest is not masked out from labels
                # make sure we create the labels first, otherwise we get incorrect lengths
                bot_labels = [IGNORE_TOKEN_ID] * len(self.bot_prefix_token_ids) + [
                    *copy.deepcopy(bot_res["input_ids"])
                ][len(self.bot_prefix_token_ids) :]

                incremental_len = len(user_res["input_ids"]) + len(bot_res["input_ids"])
                if not started_dialogue or current_len + incremental_len < self.sequence_len:
                    started_dialogue = True
                    result, current_len = parse_tokenized_to_result(
                        result,
                        current_len,
                        user_res,
                        user_labels,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                    result, current_len = parse_tokenized_to_result(
                        result,
                        current_len,
                        bot_res,
                        bot_labels,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                else:
                    LOG.warning(f"Pygmalion truncate dialogue!")
                    break
            else:
                assert False, f"unknown role in conversation: {role}"
        return result


class PygmalionPrompter:
    """
    Prompter for Pygmalion.
    """

    def __init__(self, *args, **kwargs):
        pass

    def build_prompt(
        self, source, *args, **kwargs  # pylint: disable=unused-argument
    ) -> Generator[Tuple[str, str], None, None]:
        for msg in source:
            yield msg["role"], msg["value"]


def load(tokenizer, cfg):
    return PygmalionPromptTokenizingStrategy(
        PygmalionPrompter(), tokenizer, cfg.train_on_inputs, cfg.sequence_len
    )
