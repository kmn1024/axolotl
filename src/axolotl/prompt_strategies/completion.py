"""
Basic completion text
"""
from collections import defaultdict
from typing import Any, Dict, Generator, Optional, Tuple

from axolotl.prompt_tokenizers import InstructionPromptTokenizingStrategy
from axolotl.utils.perturb_text import Perturber


class CompletionPromptTokenizingStrategy(InstructionPromptTokenizingStrategy):
    """
    Tokenizing strategy for Completion prompts.
    """

    _field: str = "text"

    def __init__(self, *args, max_length=None, **kwargs):
        super().__init__(*args, **kwargs)
        if max_length is not None:
            self.max_length = max_length
        self.perturber = Perturber.instance()

    @property
    def supports_batched(self):
        return False

    @property
    def field(self) -> str:
        return self._field

    @field.setter
    def field(self, new_field: str):
        self._field = new_field

    def parse_instruction_fields(self, prompt) -> Tuple[str, str, str]:
        return (
            prompt[self.field],
            "",
            "",
        )

    def tokenize_prompt(self, prompt):
        text_field = prompt[self._field]
        text_field = self.perturber.perturb_text(text_field)
        tokenized_full_prompt = self._tokenize(text_field)
        res = {}
        for key, val in tokenized_full_prompt.items():
            res[key] = [val[i : i + self.sequence_len] for i in range(0, len(val), self.sequence_len)]
        return res

    def _build_full_prompt(
        self, instruction, input, response
    ):  # pylint: disable=redefined-builtin
        return next(iter(self.prompter.build_prompt(instruction, input, response)))


class CompletionPrompter:
    """
    Prompter for completion
    """

    def build_prompt(
        self,
        instruction: str,
        input=None,  # pylint: disable=redefined-builtin, unused-argument
        output=None,  # pylint: disable=unused-argument
    ) -> Generator[str, None, None]:
        yield instruction


def load(tokenizer, cfg, ds_cfg: Optional[Dict[str, Any]] = None):
    strat = CompletionPromptTokenizingStrategy(
        CompletionPrompter(),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
        #max_length=cfg.sequence_len * 64,
        max_length=cfg.sequence_len * 128,
    )
    if ds_cfg and "field" in ds_cfg:
        strat.field = ds_cfg["field"]

    return strat
