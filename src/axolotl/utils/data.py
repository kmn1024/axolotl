"""Module containing data utilities"""
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import functools
import glob
import hashlib
import logging
import random
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    concatenate_datasets,
    interleave_datasets,
    load_dataset,
    load_from_disk,

)

from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerBase

from axolotl.common.const import DEFAULT_DATASET_PREPARED_PATH
from axolotl.datasets import ConstantLengthDataset, TokenizedPromptDataset
from axolotl.prompt_strategies import load
from axolotl.prompt_tokenizers import (
    AlpacaMultipleChoicePromptTokenizingStrategy,
    AlpacaPromptTokenizingStrategy,
    AlpacaReflectionPTStrategy,
    GPTeacherPromptTokenizingStrategy,
    JeopardyPromptTokenizingStrategy,
    OpenAssistantPromptTokenizingStrategy,
    SummarizeTLDRPromptTokenizingStrategy,
)
from axolotl.prompters import (
    AlpacaPrompter,
    GPTeacherPrompter,
    JeopardyPrompter,
    MultipleChoiceConcisePrompter,
    MultipleChoiceExplainPrompter,
    ReflectAlpacaPrompter,
    SummarizeTLDRPrompter,
)
from axolotl.utils.dict import DictDefault
from axolotl.utils.distributed import is_main_process, zero_first
from axolotl.utils.trainer import (
    calculate_total_num_steps,
    process_datasets_for_packing,
    streaming_process_datasets_for_packing,
)

from axolotl.utils.dataloader import MultipackDistributedDataloader, StreamingMultipackDistributedDataloader

LOG = logging.getLogger("axolotl")


def md5(to_hash: str, encoding: str = "utf-8") -> str:
    try:
        return hashlib.md5(to_hash.encode(encoding), usedforsecurity=False).hexdigest()
    except TypeError:
        return hashlib.md5(to_hash.encode(encoding)).hexdigest()  # nosec
    
def maybe_filter_columns(dataset, cfg):
    if cfg.remove_columns is not None:
        sample = next(iter(dataset['train']))
        remove_features = [c for c in cfg.remove_columns if c in sample.keys()]
        print(f"Removing: {remove_features}")
        dataset = dataset.remove_columns(remove_features)
    return dataset

def prepare_dataset(cfg, tokenizer):
    if cfg.local_streaming_datasets:
        train_dataset = load_tokenized_prepared_datasets_local_stream(tokenizer, cfg, cfg.datasets, is_eval=False)
        eval_dataset = load_tokenized_prepared_datasets_local_stream(tokenizer, cfg, cfg.eval_datasets, is_eval=True) if cfg.eval_datasets else None
        if eval_dataset:
            print(f'Eval examples: {len(eval_dataset)}')
        train_dataset, eval_dataset = streaming_process_datasets_for_packing(cfg, train_dataset, eval_dataset, tokenizer)
        return train_dataset, eval_dataset, cfg.max_steps
    elif not cfg.pretraining_dataset:
        with zero_first(is_main_process()):
            train_dataset, eval_dataset = load_prepare_datasets(
                tokenizer, cfg, cfg.datasets, DEFAULT_DATASET_PREPARED_PATH
            )
            if cfg.eval_datasets:
                assert eval_dataset is None
                eval_dataset, dummy_dataset = load_prepare_datasets(
                    tokenizer, cfg, cfg.eval_datasets, DEFAULT_DATASET_PREPARED_PATH
                )
                assert dummy_dataset is None
    else:
        train_dataset = load_pretraining_dataset(
            cfg.pretraining_dataset,
            tokenizer,
            max_tokens=cfg.sequence_len,
            seed=cfg.seed or 42,
        )
        # https://discuss.huggingface.co/t/how-to-use-huggingface-trainer-streaming-datasets-without-wrapping-it-with-torchdatas-iterablewrapper/25230
        train_dataset = train_dataset.with_format("torch")
        eval_dataset = None
        return train_dataset, eval_dataset, cfg.max_steps        

    with zero_first(is_main_process()):
        train_dataset, eval_dataset = process_datasets_for_packing(
            cfg, train_dataset, eval_dataset, tokenizer
        )
    if cfg.max_steps:
        total_num_steps = min(
            calculate_total_num_steps(cfg, train_dataset, tokenizer), cfg.max_steps
        )
        LOG.info(f"Maximum number of steps set at {total_num_steps}")
    else:
        total_num_steps = calculate_total_num_steps(cfg, train_dataset, tokenizer)
    return train_dataset, eval_dataset, total_num_steps


def postprocess_and_wrap_dataset(d, seed, ds, cfg, tokenizer, is_streaming):
    # support for using a subset of the data
    if d.shards:
        if "train" in ds:
            ds = ds.shuffle(seed=seed)["train"].shard(
                num_shards=d.shards, index=0
            )
        else:
            ds = ds.shuffle(seed=seed).shard(num_shards=d.shards, index=0)

    d_base_type = d_prompt_style = None
    d_type = d.type
    if isinstance(d_type, str):
        d_type_split = d_type.split(":")
        d_base_type = d_type_split[0]
        d_prompt_style = d_type_split[1] if len(d_type_split) > 1 else None
    if "train" in ds:
        ds = ds["train"]
    elif (
        isinstance(ds, DatasetDict)
        and d.train_on_split
        and d.train_on_split in ds
    ):
        ds = ds[d.train_on_split]
    elif isinstance(ds, DatasetDict):
        raise ValueError(
            f"no train split found for dataset {d.path}, you may specify a split with 'train_on_split: `"
        )
    if (
        not is_streaming
        and "input_ids" in ds.features
        and "attention_mask" in ds.features
        and "labels" in ds.features
    ):
        # dataset is already tokenized, just drop it straight in
        return ds
    
    ds_strategy = None
    if isinstance(d.type, DictDefault):
        ds_strategy = load("user_defined", tokenizer, cfg, d.type.to_dict())
    elif ds_strategy := load(d.type, tokenizer, cfg, d):
        pass
    elif d_base_type == "alpaca":
        ds_strategy = AlpacaPromptTokenizingStrategy(
            AlpacaPrompter(d_prompt_style),
            tokenizer,
            cfg.train_on_inputs,
            cfg.sequence_len,
        )
    elif d_base_type == "explainchoice":
        ds_strategy = AlpacaMultipleChoicePromptTokenizingStrategy(
            MultipleChoiceExplainPrompter(d_prompt_style),
            tokenizer,
            cfg.train_on_inputs,
            cfg.sequence_len,
        )
    elif d_base_type == "concisechoice":
        ds_strategy = AlpacaMultipleChoicePromptTokenizingStrategy(
            MultipleChoiceConcisePrompter(d_prompt_style),
            tokenizer,
            cfg.train_on_inputs,
            cfg.sequence_len,
        )
    elif d_base_type == "summarizetldr":
        ds_strategy = SummarizeTLDRPromptTokenizingStrategy(
            SummarizeTLDRPrompter(d_prompt_style),
            tokenizer,
            cfg.train_on_inputs,
            cfg.sequence_len,
        )
    elif d_base_type == "jeopardy":
        ds_strategy = JeopardyPromptTokenizingStrategy(
            JeopardyPrompter(d_prompt_style),
            tokenizer,
            cfg.train_on_inputs,
            cfg.sequence_len,
        )
    elif d_base_type == "oasst":
        ds_strategy = OpenAssistantPromptTokenizingStrategy(
            AlpacaPrompter(d_prompt_style),
            tokenizer,
            cfg.train_on_inputs,
            cfg.sequence_len,
        )
    elif d_base_type == "gpteacher":
        ds_strategy = GPTeacherPromptTokenizingStrategy(
            GPTeacherPrompter(d_prompt_style),
            tokenizer,
            cfg.train_on_inputs,
            cfg.sequence_len,
        )
    elif d_base_type == "reflection":
        ds_strategy = AlpacaReflectionPTStrategy(
            ReflectAlpacaPrompter(d_prompt_style),
            tokenizer,
            cfg.train_on_inputs,
            cfg.sequence_len,
        )
    else:
        suffix = ""
        if ":load_" in d.type:
            suffix = f" Did you mean {d.type.replace(':load_', '.load_')}?"
        LOG.error(f"unhandled prompt tokenization strategy: {d.type}. {suffix}")
        raise ValueError(
            f"unhandled prompt tokenization strategy: {d.type} {suffix}"
        )

    if is_streaming:
        map_kwargs = {}
        if ds_strategy.supports_batched:
            map_kwargs["batched"] = True
            map_kwargs["batch_size"] = 8
        ds = ds.map(
            ds_strategy.tokenize_prompt,
            remove_columns=cfg.dataset_columns,
            **map_kwargs,
        )
        return ds
    else:
        ds_wrapper = TokenizedPromptDataset(ds_strategy, ds, remove_columns=None)
        return ds_wrapper


def load_tokenized_prepared_datasets_local_stream(
    tokenizer, cfg, dataset_configs, is_eval 
) -> DatasetDict:
    LOG.info(f"Loading local streaming datasets... {dataset_configs}")
    if cfg.seed:
        seed = cfg.seed
    else:
        LOG.info("No seed provided, using default seed of 42")
        seed = 42
    random.seed(seed)

    def for_d_in_datasets(dataset_configs):
        for dataset in dataset_configs:
            if dataset.name and isinstance(dataset.name, list):
                for name in dataset.name:
                    yield DictDefault({**dataset, "name": name})
            else:
                yield dataset

    def load_streaming_ds(give_ds_type, ds_name, ds_fullpath):
        ds_type = "json"
        if give_ds_type:
            ds_type = give_ds_type
        elif ".parquet" in ds_fullpath:
            ds_type = "parquet"
        elif ".arrow" in ds_fullpath:
            ds_type = "arrow"
        elif ".csv" in ds_fullpath:
            ds_type = "csv"
        elif ".txt" in ds_fullpath:
            ds_type = "text"
        ds = load_dataset(
            ds_type,
            name=ds_name,
            data_files=ds_fullpath,
            streaming=False if is_eval else True,
            split=None,
        )
        ds = maybe_filter_columns(ds, cfg)
        return ds
    
    if is_eval:
        # pylint: disable=invalid-name
        datasets = []
        for d in for_d_in_datasets(dataset_configs):
            # prefer local dataset, even if hub exists
            local_path = Path(d.path)
            if local_path.exists():
                if local_path.is_dir():
                    dir_filepaths = sorted(glob.glob(os.path.join(d.path, '*')))
                    assert len(dir_filepaths) > 0
                    for dir_filepath in dir_filepaths:
                        assert os.path.isfile(dir_filepath), dir_filepath
                        file_ds_name = d.name + '_' + os.path.basename(dir_filepath)
                        datasets.append((d, load_streaming_ds(d.ds_type, file_ds_name, dir_filepath)))
                elif local_path.is_file():
                    datasets.append((d, load_streaming_ds(d.ds_type, d.name, d.path)))
                else:
                    raise ValueError(
                        "unhandled dataset load: local path exists, but is neither a directory or a file"
                    )
        datasets = [postprocess_and_wrap_dataset(d, seed, ds, cfg, tokenizer, is_streaming=True) for d, ds in datasets]
        dataset = concatenate_datasets(datasets)
    elif cfg.ordered_datasets:
        all_datasets = []
        for d in for_d_in_datasets(dataset_configs):
            datafiles = []
            # prefer local dataset, even if hub exists
            local_path = Path(d.path)
            if local_path.exists():
                if local_path.is_dir():
                    dir_filepaths = sorted(glob.glob(os.path.join(d.path, '*')))
                    assert len(dir_filepaths) > 0
                    for dir_filepath in dir_filepaths:
                        assert os.path.isfile(dir_filepath), dir_filepath
                        file_ds_name = d.name + '_' + os.path.basename(dir_filepath)
                        datafiles.append((d, file_ds_name, dir_filepath))
                elif local_path.is_file():
                    datafiles.append((d, d.name, d.path))
        
            d = None
            data_files = []
            for this_d, name, path in datafiles:
                data_files.append(path)
                if d is None:
                    d = this_d
                else:
                    assert this_d.ds_type == d.ds_type
            ds = load_dataset(
                d.ds_type,
                data_files=data_files,
                streaming=True,
                split=None,
            )
            ds = maybe_filter_columns(ds, cfg)
            ds = postprocess_and_wrap_dataset(d, seed, ds, cfg, tokenizer, is_streaming=True)
            print(f'Training data {d.name} shards: {ds.n_shards}')
            all_datasets.append(ds)
        dataset = concatenate_datasets(all_datasets)
        print(f'Training data concatenated shards: {dataset.n_shards}')
    else:
        # pylint: disable=invalid-name
        datafiles = []
        for d in for_d_in_datasets(dataset_configs):
            # prefer local dataset, even if hub exists
            local_path = Path(d.path)
            if local_path.exists():
                if local_path.is_dir():
                    dir_filepaths = sorted(glob.glob(os.path.join(d.path, '*')))
                    assert len(dir_filepaths) > 0
                    for dir_filepath in dir_filepaths:
                        assert os.path.isfile(dir_filepath), dir_filepath
                        file_ds_name = d.name + '_' + os.path.basename(dir_filepath)
                        datafiles.append((d, file_ds_name, dir_filepath))
                elif local_path.is_file():
                    datafiles.append((d, d.name, d.path))
        
        random.shuffle(datafiles)
        # For quickly resuming from checkpoint by estimating files to skip.
        num_original = len(datafiles)
        datafiles = datafiles[0:]
        print(f'Truncating datafiles: {num_original} -> {len(datafiles)}')
        d = None
        data_sets, sizes = [], []
        for this_d, name, path in datafiles:
            if d is None:
                d = this_d
            else:
                assert this_d.ds_type == d.ds_type
            ds = load_dataset(
                d.ds_type,
                data_files=path,
                streaming=True,
                split=None,
            )
            ds = maybe_filter_columns(ds, cfg)
            ds = postprocess_and_wrap_dataset(d, seed, ds, cfg, tokenizer, is_streaming=True)
            data_sets.append(ds)
            sizes.append(os.stat(path).st_size / (1024*1024))
        # Random interleave.
        probs = [one_size/sum(sizes) for one_size in sizes]
        dataset = interleave_datasets(data_sets, probabilities=probs, stopping_strategy='all_exhausted')
        #dataset = concatenate_datasets(data_sets)
        print(f'Training data shards: {dataset.n_shards}')
    return dataset


def load_tokenized_prepared_datasets(
    tokenizer, cfg, input_datasets, default_dataset_prepared_path
) -> DatasetDict:
    tokenizer_name = tokenizer.__class__.__name__
    ds_hash = str(
        md5(
            (
                str(cfg.sequence_len)
                + "@"
                + "|".join(
                    sorted([f"{d.path}:{d.type}:{d.shards}" for d in input_datasets])
                )
                + "|"
                + tokenizer_name
            )
        )
    )
    prepared_ds_path = (
        Path(cfg.dataset_prepared_path) / ds_hash
        if cfg.dataset_prepared_path
        else Path(default_dataset_prepared_path) / ds_hash
    )
    dataset = None
    use_auth_token = cfg.hf_use_auth_token
    try:
        if cfg.push_dataset_to_hub:
            dataset = load_dataset(
                f"{cfg.push_dataset_to_hub}/{ds_hash}",
                token=use_auth_token,
            )
            dataset = dataset["train"]
            dataset = maybe_filter_columns(dataset, cfg)
    except Exception:  # pylint: disable=broad-except # nosec
        pass

    if dataset:
        ...
    elif cfg.dataset_prepared_path and any(prepared_ds_path.glob("*")):
        LOG.info(f"Loading prepared dataset from disk at {prepared_ds_path}...")
        dataset = load_from_disk(str(prepared_ds_path))
        LOG.info("Prepared dataset loaded from disk...")
    else:
        LOG.info(f"Unable to find prepared dataset in {prepared_ds_path}")
        LOG.info("Loading raw datasets...")

        if cfg.seed:
            seed = cfg.seed
        else:
            LOG.info("No seed provided, using default seed of 42")
            seed = 42

        datasets = []

        def for_d_in_datasets(dataset_configs):
            for dataset in dataset_configs:
                if dataset.name and isinstance(dataset.name, list):
                    for name in dataset.name:
                        yield DictDefault({**dataset, "name": name})
                else:
                    yield dataset

        # pylint: disable=invalid-name
        for d in for_d_in_datasets(input_datasets):
            ds: Union[Dataset, DatasetDict] = None
            ds_from_hub = False
            try:
                load_dataset(
                    d.path,
                    name=d.name,
                    streaming=True,
                    token=use_auth_token,
                )
                ds_from_hub = True
            except (FileNotFoundError, ConnectionError):
                pass

            # prefer local dataset, even if hub exists
            local_path = Path(d.path)
            if local_path.exists():
                if local_path.is_dir():
                    # TODO dirs with arrow or parquet files could be loaded with `load_from_disk`
                    ds = load_dataset(
                        d.path,
                        name=d.name,
                        data_files=d.data_files,
                        streaming=False,
                        split=None,
                    )
                    ds = maybe_filter_columns(ds, cfg)
                elif local_path.is_file():
                    ds_type = "json"
                    if d.ds_type:
                        ds_type = d.ds_type
                    elif ".parquet" in d.path:
                        ds_type = "parquet"
                    elif ".arrow" in d.path:
                        ds_type = "arrow"
                    elif ".csv" in d.path:
                        ds_type = "csv"
                    elif ".txt" in d.path:
                        ds_type = "text"
                    ds = load_dataset(
                        ds_type,
                        name=d.name,
                        data_files=d.path,
                        streaming=False,
                        split=None,
                    )
                    ds = maybe_filter_columns(ds, cfg)
                else:
                    raise ValueError(
                        "unhandled dataset load: local path exists, but is neither a directory or a file"
                    )
            elif ds_from_hub:
                ds = load_dataset(
                    d.path,
                    name=d.name,
                    streaming=False,
                    data_files=d.data_files,
                    token=use_auth_token,
                )
                ds = maybe_filter_columns(ds, cfg)
            else:
                if isinstance(d.data_files, str):
                    fp = hf_hub_download(
                        repo_id=d.path,
                        repo_type="dataset",
                        filename=d.data_files,
                    )
                elif isinstance(d.data_files, list):
                    fp = []
                    for file in d.data_files:
                        fp.append(
                            hf_hub_download(
                                repo_id=d.path,
                                repo_type="dataset",
                                filename=file,
                            )
                        )
                else:
                    raise ValueError(
                        "data_files must be either a string or list of strings"
                    )
                ds = load_dataset(
                    "json", name=d.name, data_files=fp, streaming=False, split=None
                )
                ds = maybe_filter_columns(ds, cfg)
            if not ds:
                raise ValueError("unhandled dataset load")
            
            wrapped_ds = postprocess_and_wrap_dataset(d, seed, ds, cfg, tokenizer, is_streaming=False)
            if wrapped_ds:
                datasets.append(wrapped_ds)

        LOG.info("merging datasets")
        dataset = concatenate_datasets(datasets)

        if len(datasets) > 1:
            LOG.info("shuffle merged datasets")
            dataset = dataset.shuffle(seed=seed)
        if cfg.local_rank == 0:
            LOG.info(f"Saving merged prepared dataset to disk... {prepared_ds_path}")
            dataset.save_to_disk(prepared_ds_path)
            if cfg.push_dataset_to_hub:
                LOG.info(
                    f"Saving merged prepared dataset with push_to_hub... {cfg.push_dataset_to_hub}/{ds_hash}"
                )
                dataset.push_to_hub(
                    f"{cfg.push_dataset_to_hub}/{ds_hash}", private=True
                )

    return dataset


def load_prepare_datasets(
    tokenizer: PreTrainedTokenizerBase,
    cfg,
    input_datasets,
    default_dataset_prepared_path,
) -> Tuple[Dataset, Dataset]:
    max_packed_sequence_len = (
        cfg.max_packed_sequence_len if cfg.max_packed_sequence_len else cfg.sequence_len
    )
    max_packed_sequence_len = min(
        max_packed_sequence_len, cfg.sequence_len
    )  # make sure we don't accidentally set it larger than sequence_len

    tokenizer_name = tokenizer.__class__.__name__
    if cfg.max_packed_sequence_len is not None:
        # see if we can go ahead and load the stacked dataset
        seed = f"@{str(cfg.seed)}" if cfg.seed else ""
        ds_hash = str(
            md5(
                (
                    str(cfg.sequence_len)
                    + "@"
                    + str(max_packed_sequence_len)
                    + seed
                    + "|".join(
                        sorted([f"{d.path}:{d.type}:{d.shards}" for d in input_datasets])
                    )
                    + "|"
                    + tokenizer_name
                )
            )
        )
        prepared_ds_path = (
            Path(cfg.dataset_prepared_path) / ds_hash
            if cfg.dataset_prepared_path
            else Path(default_dataset_prepared_path) / ds_hash
        )

        dataset = None
        use_auth_token = cfg.hf_use_auth_token
        try:
            if cfg.push_dataset_to_hub:
                LOG.info(
                    f"Checking for packed prepared dataset from hub... {cfg.push_dataset_to_hub}/{ds_hash}"
                )
                dataset = load_dataset(
                    f"{cfg.push_dataset_to_hub}/{ds_hash}",
                    token=use_auth_token,
                )
                dataset = dataset["train"]
                dataset = maybe_filter_columns(dataset, cfg)
        except Exception:  # pylint: disable=broad-except # nosec
            pass

        if dataset:
            ...
        elif cfg.dataset_prepared_path and any(prepared_ds_path.glob("*")):
            LOG.info(
                f"Loading prepared packed dataset from disk at {prepared_ds_path}..."
            )
            dataset = load_from_disk(str(prepared_ds_path))
            LOG.info("Prepared packed dataset loaded from disk...")
            if cfg.push_dataset_to_hub:
                LOG.info(
                    f"Saving packed prepared dataset with push_to_hub... {cfg.push_dataset_to_hub}/{ds_hash}"
                )
                dataset.push_to_hub(
                    f"{cfg.push_dataset_to_hub}/{ds_hash}", private=True
                )
        else:
            dataset = load_tokenized_prepared_datasets(
                tokenizer, cfg, input_datasets, default_dataset_prepared_path
            )

            if cfg.seed:
                dataset = dataset.shuffle(seed=cfg.seed)

            constant_len_dataset = ConstantLengthDataset(
                tokenizer,
                [dataset],
                seq_length=max_packed_sequence_len,
            )
            LOG.info(f"packing master dataset to len: {cfg.max_packed_sequence_len}")
            dataset = Dataset.from_list(list(constant_len_dataset))

            # filter out bad data
            # TODO convert to dataset.filter(...)
            dataset = Dataset.from_list(
                [
                    d
                    for d in dataset
                    if len(d["input_ids"]) <= cfg.sequence_len
                    and len(d["input_ids"]) > 0
                    and len(d["input_ids"]) == len(d["attention_mask"])
                    and len(d["input_ids"]) == len(d["labels"])
                ]
            )

            if cfg.local_rank == 0:
                LOG.info(
                    f"Saving packed prepared dataset to disk... {prepared_ds_path}"
                )
                dataset.save_to_disk(prepared_ds_path)
                if cfg.push_dataset_to_hub:
                    LOG.info(
                        f"Saving packed prepared dataset with push_to_hub... {cfg.push_dataset_to_hub}/{ds_hash}"
                    )
                    dataset.push_to_hub(
                        f"{cfg.push_dataset_to_hub}/{ds_hash}",
                        private=True,
                    )
    else:
        dataset = load_tokenized_prepared_datasets(
            tokenizer, cfg, input_datasets, default_dataset_prepared_path
        )

    if cfg.dataset_shard_num and cfg.dataset_shard_idx is not None:
        LOG.info(
            f"Using index #{cfg.dataset_shard_idx} of {cfg.dataset_shard_num} shards"
        )
        dataset = dataset.shard(
            num_shards=cfg.dataset_shard_num,
            index=cfg.dataset_shard_idx,
        )

    if cfg.val_set_size:
        # ensure we end up with the same fingerprint by doing rank0 first and being able to cache
        to_hash_train = (
            dataset._fingerprint  # pylint: disable=protected-access
            + "|"
            + str(cfg.val_set_size)
            + "|"
            + "train"
            + "|"
            + str(cfg.seed or 42)
        )
        to_hash_test = (
            dataset._fingerprint  # pylint: disable=protected-access
            + "|"
            + str(cfg.val_set_size)
            + "|"
            + "test"
            + "|"
            + str(cfg.seed or 42)
        )
        train_fingerprint = md5(to_hash_train)
        test_fingerprint = md5(to_hash_test)
        with zero_first(is_main_process()):
            dataset = dataset.train_test_split(
                test_size=cfg.val_set_size,
                shuffle=False,
                seed=cfg.seed or 42,
                train_new_fingerprint=train_fingerprint,
                test_new_fingerprint=test_fingerprint,
            )
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    return train_dataset, eval_dataset


def encode_pretraining(
    tokenizer: PreTrainedTokenizerBase, max_tokens: int, examples: List[str]
) -> Dict[str, List]:
    res = tokenizer(
        examples,
        truncation=True,
        max_length=max_tokens - 2,
        add_special_tokens=True,
    )
    # Convert to PyTorch tensors
    input_ids = [torch.tensor(seq) for seq in res["input_ids"]]
    attention_mask = [torch.tensor(seq) for seq in res["attention_mask"]]
    new_input_ids = []
    new_attention_mask = []
    # Append EOS and PAD tokens to input_ids, and correct attention_mask
    for i, _ in enumerate(input_ids):
        input_ids[i] = torch.cat(
            (
                input_ids[i],
                torch.tensor([tokenizer.eos_token_id, tokenizer.pad_token_id]),
            ),
            dim=0,
        )
        attention_mask[i] = torch.cat((attention_mask[i], torch.tensor([1, 0])), dim=0)

    # Concatenate tokens so that their lengths are less than max_tokens
    buffer_input_ids = torch.tensor([], dtype=torch.long)
    buffer_attention_mask = torch.tensor([], dtype=torch.long)

    for ids, mask in zip(input_ids, attention_mask):
        if buffer_input_ids.numel() == max_tokens:
            new_input_ids.append(buffer_input_ids)
            new_attention_mask.append(buffer_attention_mask)
            buffer_input_ids = torch.tensor([], dtype=torch.long)
            buffer_attention_mask = torch.tensor([], dtype=torch.long)
            buffer_input_ids = torch.cat((buffer_input_ids, ids), dim=0)
            buffer_attention_mask = torch.cat((buffer_attention_mask, mask), dim=0)
        elif buffer_input_ids.numel() + ids.numel() <= max_tokens:
            buffer_input_ids = torch.cat((buffer_input_ids, ids), dim=0)
            buffer_attention_mask = torch.cat((buffer_attention_mask, mask), dim=0)
        else:
            buffer_input_ids = torch.cat(
                (
                    buffer_input_ids,
                    torch.full(
                        (max_tokens - buffer_input_ids.numel(),),
                        tokenizer.pad_token_id,
                        dtype=torch.long,
                    ),
                ),
                dim=0,
            )
            buffer_attention_mask = torch.cat(
                (
                    buffer_attention_mask,
                    torch.full(
                        (max_tokens - buffer_attention_mask.numel(),),
                        0,
                        dtype=torch.long,
                    ),
                ),
                dim=0,
            )
            new_input_ids.append(buffer_input_ids)
            new_attention_mask.append(buffer_attention_mask)
            buffer_input_ids = torch.tensor([], dtype=torch.long)
            buffer_attention_mask = torch.tensor([], dtype=torch.long)

            buffer_input_ids = torch.cat((buffer_input_ids, ids), dim=0)
            buffer_attention_mask = torch.cat((buffer_attention_mask, mask), dim=0)

    if buffer_input_ids.numel() > 0:  # for any leftover tokens
        while buffer_input_ids.numel() < max_tokens:  # make all sequences equal in size
            buffer_input_ids = torch.cat(
                (
                    buffer_input_ids,
                    torch.full(
                        (max_tokens - buffer_input_ids.numel(),),
                        tokenizer.pad_token_id,
                        dtype=torch.long,
                    ),
                ),
                dim=0,
            )
            buffer_attention_mask = torch.cat(
                (
                    buffer_attention_mask,
                    torch.full(
                        (max_tokens - buffer_attention_mask.numel(),),
                        0,
                        dtype=torch.long,
                    ),
                ),
                dim=0,
            )
        new_input_ids.append(buffer_input_ids)
        new_attention_mask.append(buffer_attention_mask)

    ret = {
        "input_ids": [seq.tolist() for seq in new_input_ids],
        "labels": [seq.tolist() for seq in new_input_ids],
        "attention_mask": [seq.tolist() for seq in new_attention_mask],
    }

    LOG.debug(len(ret["input_ids"]))
    return ret


def load_pretraining_dataset(path, tokenizer, max_tokens=2048, seed=42):
    encode = functools.partial(encode_pretraining, tokenizer, max_tokens)
    dataset = load_dataset(path, streaming=True, split="train")
    dataset = dataset.shuffle(seed=seed, buffer_size=10_000)
    dataset = dataset.map(
        encode,
        batched=True,
        input_columns="text",
        # remove all the existing columns after mapping since they end up having
        # a different length than the encoded/tokenized column
        remove_columns=dataset.features.keys(),
    )
    return dataset
