import ollama
from tqdm import tqdm



def analyze_emo(text, model_name):
    message = f"""
    Analyze the emotion valence of the following sentence and explain it. 
    Sentence: {text}
    Output format:
    Emotion analysis reason: <brief reason>
    """
    res = ollama.chat(model=model_name, stream=False, messages=[{"role": "user", "content": message}],
                      options={"temperature": 0})
    result_content = res["message"]["content"]
    emotion_analysis_reason = result_content.split("Emotion analysis reason:")[
        1].strip() if "Emotion analysis reason:" in result_content else "none"
    return emotion_analysis_reason


def target_source_domains(text, emotion_analysis_reason, model_name):
    message = f"""
    Based on emotion valence, identify the target and source domains in the following sentence.
    If no metaphor can be identified, set all fields to 'none'.
    Sentence: {text}
    Emotion analysis reason: {emotion_analysis_reason}
    Output format:
    Target domain: <target domain or 'none'>
    Select reason: <reason or 'none'>
    Source domain: <source domain or 'none'>
    Generate reason: <reason or 'none'>
    """
    res = ollama.chat(model=model_name, stream=False, messages=[{"role": "user", "content": message}],
                      options={"temperature": 0})
    result_content = res["message"]["content"]
    target_domain = result_content.split("Target domain:")[1].split("\n")[
        0].strip() if "Target domain:" in result_content else "none"
    select_reason = result_content.split("Select reason:")[1].split("\n")[
        0].strip() if "Select reason:" in result_content else "none"
    source_domain = result_content.split("Source domain:")[1].split("\n")[
        0].strip() if "Source domain:" in result_content else "none"
    generate_reason = result_content.split("Generate reason:")[
        1].strip() if "Generate reason:" in result_content else "none"
    return target_domain, select_reason, source_domain, generate_reason


def metaphor_learning(text, emotion_analysis_reason, target_domain, source_domain, model_name):
    message = f"""
    Re-evaluate if the following sentence contains metaphor, based on previous judgement, the emotion and domain knowledge.
    Sentence: {text}
    Emotion analysis reason: {emotion_analysis_reason}
    Target domain: {target_domain}
    Source domain: {source_domain}
    Output format:
    Metaphor judgment: <0 or 1>
    Metaphor reason: <brief reason>
    """
    res = ollama.chat(model=model_name, stream=False, messages=[{"role": "user", "content": message}],
                      options={"temperature": 0})
    result_content = res["message"]["content"]


    metaphor_judgment = safe_parse_int(result_content.split("Metaphor judgment:")[1].split("\n")[0])
    metaphor_reason = result_content.split("Metaphor reason:")[
        1].strip() if "Metaphor reason:" in result_content else "none"

    return metaphor_judgment, metaphor_reason


def hyperbole_learning(text, metaphor_judgment, emotion_analysis_reason, target_domain,
                                              source_domain, model_name):
    message = f"""
    Based on previous judgement and knowledge, re-evaluate if the sentence contains hyperbole.
    Sentence: {text}
    Metaphor judgment: {metaphor_judgment}
    Emotion analysis reason: {emotion_analysis_reason}
    Target domain: {target_domain}
    Source domain: {source_domain}
    Output format:
    Hyperbole judgment: <0 or 1>
    Hyperbole reason: <brief reason>
    """
    res = ollama.chat(model=model_name, stream=False, messages=[{"role": "user", "content": message}],
                      options={"temperature": 0})
    result_content = res["message"]["content"]

    hyperbole_judgment = safe_parse_int(
        result_content.split("Hyperbole judgment:")[1].split("\n")[0]
    ) if "Hyperbole judgment:" in result_content else 0

    hyperbole_reason = result_content.split("Hyperbole reason:")[
        1].strip() if "Hyperbole reason:" in result_content else "none"

    return hyperbole_judgment, hyperbole_reason


def hyperbole_metaphor(text, emotion_analysis_reason, hyperbole_judgment, hyperbole_reason, metaphor_judgment,
                             metaphor_reason, target_domain, source_domain, model_name):
    message = f"""
    You are a LLM with rich metaphor and hyperbole knowledge and strong judgment and reasoning ability. Please re-verify according to previous knowledge, including previous metaphor and exaggeration judgment and corresponding judgment reason, emotion and domain knowledge, to determine whether the current input is wrong. If your prediction is wrong, re-reason.
    Sentence: {text}
    Emotion analysis reason: {emotion_analysis_reason}
    Target domain: {target_domain}
    Source domain: {source_domain}
    Previous hyperbole judgment: {hyperbole_judgment}
    Previous metaphor judgment: {metaphor_judgment}
    Previous hyperbole reason: {hyperbole_reason}
    Previous metaphor reason: {metaphor_reason}
    Output format:
    Hyperbole judgment: <0 or 1>
    Hyperbole reason: <brief reason>
    Metaphor judgment: <0 or 1>
    Metaphor reason: <brief reason>
    """
    res = ollama.chat(model=model_name, stream=False, messages=[{"role": "user", "content": message}],
                      options={"temperature": 0})
    result_content = res["message"]["content"]


    def safe_parse_int(text, default=0):
        cleaned_text = re.sub(r'[^\d]', '', text)
        return int(cleaned_text) if cleaned_text.isdigit() else default

    hyperbole_judgment = safe_parse_int(
        result_content.split("Hyperbole judgment:")[1].split("\n")[0]) if "Hyperbole judgment:" in result_content else 0
    hyperbole_reason = result_content.split("Hyperbole reason:")[1].split("\n")[
        0].strip() if "Hyperbole reason:" in result_content else "none"
    metaphor_judgment = safe_parse_int(
        result_content.split("Metaphor judgment:")[1].split("\n")[0]) if "Metaphor judgment:" in result_content else 0
    metaphor_reason = result_content.split("Metaphor reason:")[
        1].strip() if "Metaphor reason:" in result_content else "none"

    return hyperbole_judgment, hyperbole_reason, metaphor_judgment, metaphor_reason



class NLPDataCollator:


    def __call__(
        self, features: List[Union[InputDataClass, Dict]]
    ) -> Dict[str, torch.Tensor]:
        first = features[0]
        if isinstance(first, dict):
            # NLP data sets current works presents features as lists of dictionary
            # (one per example), so we  will adapt the collate_batch logic for that
            if "labels" in first and first["labels"] is not None:
                if first["labels"].dtype == torch.int64:
                    labels = torch.tensor(
                        [f["labels"] for f in features], dtype=torch.long
                    )
                else:
                    labels = torch.tensor(
                        [f["labels"] for f in features], dtype=torch.float
                    )
                batch = {"labels": labels}
            for k, v in first.items():
                if k != "labels" and v is not None and not isinstance(v, str):
                    batch[k] = torch.stack([f[k] for f in features])
            return batch
        else:
            # otherwise, revert to using the default collate_batch
            return DefaultDataCollator().collate_batch(features)


class StrIgnoreDevice(str):

    def to(self, device):
        return self


class DataLoaderWithTaskname:

    def __init__(self, task_name, data_loader):
        self.task_name = task_name
        self.data_loader = data_loader

        self.batch_size = data_loader.batch_size
        self.dataset = data_loader.dataset

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        for batch in self.data_loader:
            batch["task_name"] = StrIgnoreDevice(self.task_name)
            yield batch


class MultitaskDataloader:

    def __init__(self, dataloader_dict):
        self.dataloader_dict = dataloader_dict
        self.num_batches_dict = {
            task_name: len(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        self.task_name_list = list(self.dataloader_dict)
        self.dataset = [None] * sum(
            len(dataloader.dataset) for dataloader in self.dataloader_dict.values()
        )

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):

        task_choice_list = []
        for i, task_name in enumerate(self.task_name_list):
            task_choice_list += [i] * self.num_batches_dict[task_name]
        task_choice_list = np.array(task_choice_list)
        np.random.shuffle(task_choice_list)
        dataloader_iter_dict = {
            task_name: iter(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        for task_choice in task_choice_list:
            task_name = self.task_name_list[task_choice]
            yield next(dataloader_iter_dict[task_name])


class MultitaskTrainer(transformers.Trainer):
    def get_single_train_dataloader(self, task_name, train_dataset):

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_sampler = (
            RandomSampler(train_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(train_dataset)
        )

        data_loader = DataLoaderWithTaskname(
            task_name=task_name,
            data_loader=DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator,
            ),
        )
        return data_loader

    def get_train_dataloader(self):

        return MultitaskDataloader(
            {
                task_name: self.get_single_train_dataloader(task_name, task_dataset)
                for task_name, task_dataset in self.train_dataset.items()
            }
        )

class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict):
        """
        Setting MultitaskModel up as a PretrainedModel allows us
        to take better advantage of Trainer features
        """
        super().__init__(transformers.PretrainedConfig())

        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)


    def create(cls, model_name, model_type_dict, model_config_dict):

        shared_encoder = None
        taskmodels_dict = {}
        for task_name, model_type in model_type_dict.items():
            model = model_type.from_pretrained(
                model_name,
                config=model_config_dict[task_name],
            )
            if shared_encoder is None:
                shared_encoder = getattr(model, cls.get_encoder_attr_name(model))
            else:
                setattr(model, cls.get_encoder_attr_name(model), shared_encoder)
            taskmodels_dict[task_name] = model
        return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)


    def get_encoder_attr_name(cls, model):
        """
        The encoder transformer is named differently in each model "architecture".
        This method lets us get the name of the encoder attribute
        """
        model_class_name = model.__class__.__name__
        if model_class_name.startswith("Bert"):
            return "bert"
        elif model_class_name.startswith("Roberta"):
            return "roberta"
        elif model_class_name.startswith("Albert"):
            return "albert"
        else:
            raise KeyError(f"Add support for new model {model_class_name}")

    def forward(self, task_name, **kwargs):
        return self.taskmodels_dict[task_name](**kwargs)
