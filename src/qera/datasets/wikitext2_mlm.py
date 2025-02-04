from itertools import chain
import datasets as hf_datasets


def get_raw_data_module_wikitext2_mlm() -> hf_datasets.DatasetDict:
    dataset_dict = hf_datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    return dataset_dict


def preprocess_data_module_wikitext2_mlm(
    raw_dataset_dict: hf_datasets.DatasetDict,
    tokenizer,
    max_length: int,  # block_size
    num_proc: int,
    overwrite_cache: bool = False,
):
    column_names = raw_dataset_dict["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

    tokenized_datasets = raw_dataset_dict.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=column_names,
        load_from_cache_file=not overwrite_cache,
        desc="Running tokenizer on every text in dataset",
    )

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // max_length) * max_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    tokenized_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=not overwrite_cache,
        desc=f"Grouping texts in chunks of {max_length}",
    )

    return tokenized_datasets
