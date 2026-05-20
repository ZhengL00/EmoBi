MODEL_NAME = "llama3:8b"
RESULTS_DIR = ".."

class DatasetConfig(datasets.BuilderConfig):

    def __init__(self, **kwargs):

        super(DatasetConfig, self).__init__(**kwargs)


class MultitaskDataset(datasets.GeneratorBasedBuilder):
    """MultitaskDataset: Version 1.0.0"""

    BUILDER_CONFIGS = [
        DatasetConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="Multitask dataset",
            features=datasets.Features(
                {
                    "doc": datasets.Value("string"),
                    "target": datasets.Value("int32"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="",
            citation="",
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(self.config.data_files)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": downloaded_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": downloaded_files["validation"]},
            ),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        data = None
        if ".tsv" in filepath:
            data = pd.read_csv(filepath, sep="\t")
        else:
            print("\nFilepath : ",filepath[0])
            data = pd.read_csv(filepath[0])
        data = data
        for idx, row in data.iterrows():
            yield row["id"], {
                "doc": row["sentence"],
                "target": row["label"],
            }

def convert_to_features(example_batch, model_name="roberta-large", max_length=128):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    inputs = list(example_batch["doc"])

    features = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    features["labels"] = example_batch["target"]
    return features
