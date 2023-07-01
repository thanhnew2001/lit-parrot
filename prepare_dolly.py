

DATA_FILE = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
DATA_FILE_NAME = "dolly_data_cleaned_archive.json"

def prepare(
    destination_path: Path = Path("data/dolly"),
    checkpoint_dir: Path = Path("checkpoints/togethercomputer/RedPajama-INCITE-Base-3B-v1"),
    test_split_size: int = 2000,
    max_seq_length: int = 256,
    seed: int = 42,
    mask_inputs: bool = False,  # as in alpaca-lora
    data_file_name: str = DATA_FILE_NAME,
) -> None:
    """Prepare the Dolly dataset for instruction tuning.

    The output is a training and validation dataset saved as `train.pt` and `val.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    destination_path.mkdir(parents=True, exist_ok=True)
    file_path = destination_path / data_file_name
    download(file_path)

    tokenizer = Tokenizer(checkpoint_dir / "tokenizer.json", checkpoint_dir / "tokenizer_config.json")

    with open(file_path, "r") as file:
        data = file.readlines()
        data = [json.loads(line) for line in data]
    for item in data:
        item["input"] = item.pop("context")
        item["output"] = item.pop("response")

    # Partition the dataset into train and test
    train_split_size = len(data) - test_split_size
    train_set, test_set = random_split(
        data, lengths=(train_split_size, test_split_size), generator=torch.Generator().manual_seed(seed)
    )
    train_set, test_set = list(train_set), list(test_set)

    print(f"train has {len(train_set):,} samples")
    print(f"val has {len(test_set):,} samples")

    print("Processing train split ...")
    train_set = [prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) for sample in tqdm(train_set)]
    torch.save(train_set, file_path.parent / "train.pt")

    print("Processing test split ...")
    test_set = [prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) for sample in tqdm(test_set)]
    torch.save(test_set, file_path.parent / "test.pt")
