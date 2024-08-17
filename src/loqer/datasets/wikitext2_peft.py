from itertools import chain
import datasets as hf_datasets


def get_raw_data_module_wikitext2_peft() -> hf_datasets.DatasetDict:
    dataset_dict = hf_datasets.load_dataset("wikitext", "wikitext-2-raw-v1")
    return dataset_dict


def preprocess_data_module_wikitext2_peft(
    raw_dataset_dict,
    tokenizer,
    max_length: int,  # block_size
    num_proc: int,
    overwrite_cache: bool = False,
) -> hf_datasets.DatasetDict:

    def tokenizer_function(examples):
        return tokenizer(examples["text"])

    tokenized_dataset_dict = raw_dataset_dict.map(
        tokenizer_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=raw_dataset_dict["train"].column_names,
        load_from_cache_file=not overwrite_cache,
        desc="Tokenizing dataset",
    )
    block_size = max_length

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    preprocessed_dataset_dict = tokenized_dataset_dict.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=not overwrite_cache,
        desc="Grouping texts",
    )

    return preprocessed_dataset_dict
