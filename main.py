import os
from tqdm import tqdm
from utils import read_csv, store_data
from model import analyze_emo, target_source_domains, metaphor_learning, \
    hyperbole_learning, hyperbole_metaphor
from config import MODEL_NAME, RESULTS_DIR


def main():
    data_dir = ".."
    input_name = ".."

    pf = read_csv(data_dir, input_name)
    emo, domain, meta, hype, result = [], [], [], [], []

    for i, row in tqdm(pf.iterrows(), total=len(pf)):
        text = row["Sentence"]
        true_hyperbole = int(row["Hyperbole"]) if not pd.isna(row["Hyperbole"]) else 0
        true_metaphor = int(row["Metaphor"]) if not pd.isna(row["Metaphor"]) else 0

        emotion_analysis_reason = analyze_emo(text, MODEL_NAME)
        emo.append({"Sentence": text, "Emotion analysis reason": emotion_analysis_reason})

        target_domain, select_reason, source_domain, generate_reason = target_source_domains(
            text, emotion_analysis_reason, MODEL_NAME
        )
        domain.append({
            "Sentence": text,
            "Target domain": target_domain,
            "Select reason": select_reason,
            "Source domain": source_domain,
            "Generate reason": generate_reason
        })

        metaphor_judgment, metaphor_reason = metaphor_learning(text, emotion_analysis_reason,
                                                                             target_domain, source_domain, MODEL_NAME)
        meta.append({
            "Sentence": text,
            "Metaphor judgment": metaphor_judgment,
            "Metaphor reason": metaphor_reason
        })

        hyperbole_judgment, hyperbole_reason = hyperbole_learning(text, metaphor_judgment,
                                                                                         emotion_analysis_reason,
                                                                                         target_domain, source_domain,
                                                                                         MODEL_NAME)
        hype.append({
            "Sentence": text,
            "Metaphor judgment": metaphor_judgment,
            "Metaphor reason": metaphor_reason,
            "Hyperbole judgment": hyperbole_judgment,
            "Hyperbole reason": hyperbole_reason
        })


        hyperbole_judgment, hyperbole_reason, metaphor_judgment, metaphor_reason = hyperbole_metaphor(
            text, emotion_analysis_reason, hyperbole_judgment, hyperbole_reason, metaphor_judgment, metaphor_reason,
            target_domain, source_domain, MODEL_NAME
        )
        result.append({
            "Sentence": text,
            "Hyperbole judgment": hyperbole_judgment,
            "Hyperbole reason": hyperbole_reason,
            "Metaphor judgment": metaphor_judgment,
            "Metaphor reason": metaphor_reason
        })

def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task"
    )
    parser.add_argument(
        "--dataset_config_name",
        type=None,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--htrain_file",
        type=None,
        default=None,
        help="A csv or a json file containing the training data.",
    )
    parser.add_argument(
        "--hvalidation_file",
        type=None,
        default=None,
        help="A csv or a json file containing the validation data.",
    )
    parser.add_argument(
        "--mtrain_file",
        type=None,
        default=None,
        help="A csv or a json file containing the training data.",
    )
    parser.add_argument(
        "--mvalidation_file",
        type=str,
        default=None,
        help="A csv or a json file containing the validation data.",
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to "
        "padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text " "(useful for T5 models).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=4,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache",
        type=bool,
        default=None,
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help="The maximum total sequence length for validation "
        "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
        "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
        "param of ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="Number of beams to use for evaluation. This argument will be "
        "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the full texts (for summarization).",
    )
    parser.add_argument(
        "--summary_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the summaries (for summarization).",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=10,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )

    args = parser.parse_args()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args

if __name__ == "__main__":
    main()
