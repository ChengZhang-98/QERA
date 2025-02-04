import datasets as hf_datasets
from datasets import load_dataset
import copy

def get_raw_data_module_gsm8k() -> hf_datasets.DatasetDict:
    dataset_dict = hf_datasets.load_dataset('gsm8k', 'main')
    return dataset_dict


def preprocess_data_module_gsm8k(
    raw_dataset_dict,
    tokenizer,
    args,
) -> hf_datasets.DatasetDict:
    if args.dataset_name and raw_dataset_dict is not None:
        # Downloading and loading a dataset from the hub.
        if "validation" not in raw_dataset_dict.keys():
            raw_dataset_dict["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_dataset_dict["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_dataset_dict = load_dataset(extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_dataset_dict.keys():
            raw_dataset_dict["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_dataset_dict["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    ##########################
    #      GSM8K dataset     #
    ##########################

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_dataset_dict["train"].column_names

    # Get the column names for source/target.
    source_column, target_column = "question", "answer"

    # Temporarily set max_target_length for training.
    padding = "max_length" if args.pad_to_max_length else False
    task_prompt = "\nAnswer the above question. First think step by step and then answer the final number.\n"

    def prompt_process(sent_1, sent_2, prompt_1="", prompt_2="", prompt_3=""):
        sent_2 = sent_2.replace("####", "The final answer is")
        return prompt_1 + sent_1 + prompt_2 + sent_2 + prompt_3

    def preprocess_function_train(examples):
        sources = examples[source_column]
        targets = examples[target_column]

        inputs = [prompt_process(source, target, prompt_2=task_prompt) for (source, target) in zip(sources, targets)]

        model_inputs = tokenizer(
            inputs,
            max_length=args.max_source_length + args.max_target_length,
            padding=padding,
            truncation=True,
            return_tensors="pt",
        )

        labels = copy.deepcopy(model_inputs)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            # get the length of the target tokens. -1 to kick out the <BOS> token
            target_tokens = tokenizer(targets, padding=False)
            target_len = [len(label) - 1 for label in target_tokens["input_ids"]]

            # don't calculate the loss from source and padding (left padding)
            for i in range(len(labels["input_ids"])):
                labels["input_ids"][i, : -target_len[i]] = -100

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def preprocess_function_test(examples):
        sources = examples[source_column]
        labels = examples[target_column]

        inputs = [source + task_prompt for source in sources]

        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)
        labels = tokenizer(labels, max_length=args.max_target_length, padding=padding, truncation=True)

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    # with accelerator.main_process_first():
    train_dataset = raw_dataset_dict["train"].map(
        preprocess_function_train,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on training dataset",
    )

    eval_dataset = raw_dataset_dict["test"].map(
        preprocess_function_test,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on test dataset",
    )

    return hf_datasets.DatasetDict(train=train_dataset, validation=eval_dataset)
