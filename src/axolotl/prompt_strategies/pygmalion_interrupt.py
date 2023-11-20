"""Module containing the PygmalionInterruptPromptTokenizingStrategy and PygmalionPrompter class"""

import copy
import logging
import random
from collections import defaultdict
from typing import Generator, List, Tuple
from axolotl.utils.perturb_text import Perturber

from axolotl.prompt_tokenizers import (
    PromptTokenizingStrategy,
    parse_tokenized_to_result,
    tokenize_prompt_default,
)

LOG = logging.getLogger("axolotl")

IGNORE_TOKEN_ID = -100
RANDOM_INJECT_SYSTEM_RATE = 0.2

class PygmalionInterruptPromptTokenizingStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for Pygmalion.
    """

    def __init__(self, prompter, tokenizer, *args, **kwargs):
        super().__init__(prompter, tokenizer, *args, **kwargs)
        res = self._tokenize("<|model|>", add_eos_token=False, strip_bos_token=True)
        bot_prefix_token_ids = res["input_ids"]
        assert len(bot_prefix_token_ids) == 1, bot_prefix_token_ids
        res = self._tokenize("<|user|>", add_eos_token=False, strip_bos_token=True)
        user_prefix_token_ids = res["input_ids"]
        assert len(user_prefix_token_ids) == 1, user_prefix_token_ids
        self.perturber = Perturber.instance()

    @property
    def supports_batched(self):
        return True

    def tokenize_prompt(self, prompt):
        INCREMENT = int(0.7 * self.sequence_len)
        chunked_result, _ = tokenize_prompt_default()

        for one_conversation_batch in prompt["conversations"]:
            conversations = self.prompter.build_prompt(one_conversation_batch)
            result, current_len = tokenize_prompt_default()
            system_res, system_labels = None, None
            prev_message_is_system = False
    
            while (next_item := next(conversations, None)) is not None:
                role, message = next_item
                if role == "system":
                    prev_message_is_system = True
                    prefix = "<|system|>"
                    # this should include a bos token, no eos token, strip trailing "\n<START>"
                    if message.endswith("\n<START>"):
                        message = message[:-8]
                    system_res = self._tokenize(
                        prefix + "Persona: " + message.strip(),
                        add_eos_token=False,
                        strip_bos_token=False,
                    )
                    # everything from this is masked out from the labels
                    system_labels = [IGNORE_TOKEN_ID] * len(system_res["input_ids"])
                    # pylint: disable=duplicate-code
                    result, current_len = parse_tokenized_to_result(
                        result,
                        current_len,
                        system_res,
                        system_labels,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                elif role == "human":
                    if system_res and not prev_message_is_system and random.random() <= RANDOM_INJECT_SYSTEM_RATE:
                        prev_message_is_system = True
                        result, current_len = parse_tokenized_to_result(
                            result,
                            current_len,
                            system_res,
                            system_labels,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )

                    prev_message_is_system = False
                    next_next_item = next(conversations, None)
                    assert next_next_item is not None
                    bot_role, bot_message = next_next_item
                    assert bot_role == "bot"
                
                    user_prefix = "<|user|>"
                    user_message = self.perturber.perturb_text(message.strip())
                    user_res = self._tokenize(
                        user_prefix + " " + user_message,
                        add_eos_token=False,
                        strip_bos_token=True,
                    )
                    # Number of tokens that can be used for training before the bot message.
                    # The user_prefix is not trainable, but the bot_prefix is (i.e. when should bot speak).
                    # We want to skip at least the first 2 tokens of user_res (excluding user_prefix), so that bot
                    # learns to listen at least to a bit of user input, before potentially interrupting.
                    assert len(user_res["input_ids"]) >= 1, f'{user_prefix + " " + user_message} -> {user_res}'
                    trainable_user_tokens = len(user_res["input_ids"]) - 1
                    trainable_tokens_start = random.randint(min(2, trainable_user_tokens), trainable_user_tokens + 1)
                    # everything up to ignored_user_tokens_end is masked out and not trained on.
                    ignored_user_tokens_end = (1 + min(trainable_tokens_start, trainable_user_tokens))
                    user_labels = ([IGNORE_TOKEN_ID] * ignored_user_tokens_end +
                                   [*copy.deepcopy(user_res["input_ids"])][ignored_user_tokens_end:])

                    bot_prefix = "<|model|>"
                    bot_res = self._tokenize(
                        bot_prefix + " " + bot_message.strip(),
                        add_eos_token=True,
                        strip_bos_token=True,
                    )
                    # mask out the prefix token, rest is not masked out from labels
                    # make sure we create the labels first, otherwise we get incorrect lengths
                    if trainable_tokens_start == trainable_user_tokens + 1:
                        assert user_labels == [IGNORE_TOKEN_ID] * len(user_res["input_ids"]), f'user_res: {user_res} | trainable_user_tokens:{trainable_user_tokens} | trainable_tokens_start:{trainable_tokens_start}'
                        bot_labels = [IGNORE_TOKEN_ID] + [*copy.deepcopy(bot_res["input_ids"])][1:]
                    else:
                        bot_labels = [*copy.deepcopy(bot_res["input_ids"])]

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
                    assert False, f"unknown role in conversation: {role}"
        
            # Else, need to chunk into multiple examples. The overlapping tokens must be masked.
            for key, val in result.items():
                if key == 'labels':
                    mask = IGNORE_TOKEN_ID
                else:
                    assert key in ['input_ids', 'attention_mask'], key
                    mask = None

                previous_end = 0
                for i in range(0, len(val), INCREMENT):
                    chunked_result[key].append(val[i : i + self.sequence_len])
                    if mask and previous_end > 0:
                        mask_range = min(previous_end - i, len(chunked_result[key][-1]))
                        assert mask_range > 0, f'{previous_end} : {i} from {len(val)}'
                        chunked_result[key][-1][:mask_range] = [mask]*mask_range
                    previous_end = i + self.sequence_len
        return chunked_result


class PygmalionInterruptPrompter:
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
    return PygmalionInterruptPromptTokenizingStrategy(
        PygmalionInterruptPrompter(), tokenizer, cfg.train_on_inputs, cfg.sequence_len
    )
