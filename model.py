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
        self.data = data_loader.data

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
        model_class_name = model.__class__.__name__
        if model_class_name.startswith("Bert"):
            return "bert"
        elif model_class_name.startswith("Roberta"):
            return "roberta"
        elif model_class_name.startswith("Albert"):
            return "albert"
        elif model_class_name.startswith("T5"):
            return "T5"
        else:
            raise KeyError(f"Add support for new model {model_class_name}")

    def forward(self, task_name, **kwargs):
        return self.taskmodels_dict[task_name](**kwargs)


class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout=0.2):
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x

class MlpSigmoid(nn.Module):
    def __init__(self, n_in, n_out, dropout=0.2):
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x



class EnhancedLSTM(torch.nn.Module):

    def __init__(self,
                 lstm_type,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 ff_dropout: float = 0.0,
                 recurrent_dropout: float = 0.0,
                 bidirectional=True) -> None:
        super().__init__()

        self.lstm_type = lstm_type

        if lstm_type == "allen":
            from AllenNLPCode.custom_stacked_bidirectional_lstm import CustomStackedBidirectionalLstm
            self.provider = CustomStackedBidirectionalLstm(
                input_size, hidden_size, num_layers, ff_dropout,
                recurrent_dropout)
        elif lstm_type == "drop_connect":
            self.provider = WeightDropLSTM(
                input_size,
                hidden_size,
                num_layers,
                ff_dropout,
                recurrent_dropout,
                bidirectional=bidirectional)
        elif lstm_type == "native":
            self.provider = torch.nn.LSTM(
                input_size,
                hidden_size,
                num_layers=num_layers,
                dropout=0,
                bidirectional=bidirectional,
                batch_first=True)
        else:
            raise Exception(lstm_type + " is an invalid lstm type")

    def forward(self, inputs, hidden, lengths):
        seq_len = inputs.shape[1]
        if self.lstm_type in ["allen", "native"]:
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                inputs, lengths, batch_first=True)

            output, _ = self.provider(packed, hidden)

            output, _ = torch.nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True)

            return output
        elif self.lstm_type == "drop_connect":
            return self.provider(inputs, lengths, seq_len)


class WeightDropLSTM(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 ff_dropout: float = 0.0,
                 recurrent_dropout: float = 0.0,
                 bidirectional=True) -> None:
        super().__init__()

        self.locked_dropout = LockedDropout()
        self.lstms = [
            torch.nn.LSTM(
                input_size
                if l == 0 else hidden_size * (1 + int(bidirectional)),
                hidden_size,
                num_layers=1,
                dropout=0,
                bidirectional=bidirectional,
                batch_first=True) for l in range(num_layers)
        ]
        if recurrent_dropout:
            self.lstms = [
                WeightDrop(lstm, ['weight_hh_l0'], dropout=recurrent_dropout)
                for lstm in self.lstms
            ]

        self.lstms = torch.nn.ModuleList(self.lstms)
        self.ff_dropout = ff_dropout
        self.num_layers = num_layers

    def forward(self, input, lengths, seq_len):
        output = input
        for lstm in self.lstms:
            output = self.locked_dropout(
                output, batch_first=True, p=self.ff_dropout)
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                output, lengths, batch_first=True, enforce_sorted=False)
            output, _ = lstm(packed, None)
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True, total_length=seq_len)

        return output

class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, batch_first=False, p=0.5):
        if not self.training or not p:
            return x
        mask_shape = (x.size(0), 1, x.size(2)) if batch_first else (1,
                                                                    x.size(1),
                                                                    x.size(2))

        mask = x.data.new(*mask_shape).bernoulli_(1 - p).div_(1 - p)
        return mask * x

class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        for name_w in weights:
            w = getattr(module, name_w)
            self.register_parameter(name_w + '_raw', Parameter(w.data))

    def _compute_dropped_weights(self):
        weight_dict = {}
        for name_w in self.weights:
            raw_w = getattr(self, name_w + '_raw')
            mask = torch.ones(1, raw_w.size(1), device=raw_w.device)
            mask = torch.nn.functional.dropout(mask, p=self.dropout, training=self.training)
            w = mask.expand_as(raw_w) * raw_w
            w = torch.nn.Parameter(w, requires_grad=False) 
            weight_dict[name_w] = w
        return weight_dict

    def forward(self, *args):
        if self.training and self.dropout > 0:
            weights = self._compute_dropped_weights()
            for name, w in weights.items():
                setattr(self.module, name, w)
        output = self.module(*args)
        return output

def init_esim_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
    if isinstance(module, nn.Embedding):
        nn.init.uniform_(module.weight, -0.1, 0.1)

class Transformer(nn.Module):
    def __init__(self, d_model, num_layers=1, nhead=1, dropout=0.1, dim_feedforward=128, max_seq_length=5000):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.pos_encoder = nn.Embedding(max_seq_length, d_model)
        self.encoder = TransformerEncoder(TransformerLayer(d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout), num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input, attention_mask=None):
        seq_length = input.size()[1]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input.device)
        positions_embedding = self.pos_encoder(position_ids).unsqueeze(0).expand(input.size())
        input = input + positions_embedding
        input = self.norm(input)
        hidden = self.encoder(input, attention_mask=attention_mask)
        out = self.decoder(hidden)
        out = (out[:,0,:], out, hidden)
        return out



class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, nhead=1, dim_feedforward=128, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.self_attention = Attention(hidden_size, nhead, dropout)
        self.fc = nn.Sequential(nn.Linear(hidden_size, dim_feedforward), nn.ReLU(), nn.Linear(dim_feedforward, hidden_size))
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, attention_mask=None):
        src_1 = self.self_attention(src, src, attention_mask=attention_mask)
        src = src + self.dropout1(src_1)
        src = self.norm1(src)
        src_2 = self.fc(src)
        src = src + self.dropout2(src_2)
        src = self.norm2(src)

        return src


class TransformerEncoder(nn.Module):
    def __init__(self, layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(layer, num_layers)
    def forward(self, src, attention_mask=None):
        for layer in self.layers:
            new_src = layer(src, attention_mask=attention_mask)
            src = src + new_src
        return src

class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, ctx_dim=None):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if ctx_dim is None:
            ctx_dim = hidden_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                batch_size, seq_len_mask = attention_mask.shape
                seq_len_k = attention_scores.shape[-1]

                if seq_len_mask != seq_len_k:
                    if seq_len_mask < seq_len_k:
                        pad_size = seq_len_k - seq_len_mask
                        attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_size), value=float('-inf'))
                    else:
                        attention_mask = attention_mask[:, :seq_len_k]

                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 4:
                pass
            else:
                raise ValueError("Unsupported attention mask dimension")

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class AttentionOutput(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(AttentionOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class Gate(nn.Module):
    def __init__(self, in_sz, out_sz):
        super(Gate, self).__init__()

        self.W_1 = nn.Parameter(torch.Tensor(in_sz, out_sz))
        self.W_2 = nn.Parameter(torch.Tensor(in_sz, out_sz))
        self.b = nn.Parameter(torch.Tensor(out_sz))
        self.init_weights()

    def forward(self, hidden_states1, hidden_states2):
        G = torch.sigmoid(hidden_states1 @ self.W_1 + hidden_states2 @ self.W_2 + self.b)
        Z = G * hidden_states1 + (1-G) * hidden_states2
        return Z

    def init_weights(self):
        init.kaiming_uniform_(self.W_1, a=math.sqrt(5))
        init.kaiming_uniform_(self.W_2, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.W_1)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.b, -bound, bound)

class BiAttention(nn.Module):
    def __init__(self, hidden_size, cross_size1, nhead=1, dropout=0.1):
        super(BiAttention, self).__init__()
        self.cross_attention_1 = Attention(hidden_size, nhead, dropout, ctx_dim=cross_size1)
        self.self_attention = Attention(hidden_size, nhead, dropout)

        self.out1 = AttentionOutput(hidden_size, dropout)
        self.out2 = AttentionOutput(hidden_size, dropout)

    def forward(self, hidden_states, cross_states_1, attention_mask=None):
        seq_len = hidden_states.size(1)

        cross_states_1 = cross_states_1.unsqueeze(1).repeat(1, seq_len, 1)
        cross_fusion = self.cross_attention_1(hidden_states, cross_states_1, attention_mask=attention_mask)
        cross_fusion = self.out1(cross_fusion, hidden_states)

        self_out = self.self_attention(cross_fusion, cross_fusion, attention_mask=attention_mask)
        self_out = self.out2(self_out, cross_fusion)

        return self_out

class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_size, cross_size1, cross_size2, nhead=1, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()
        self.cross_attention_1 = Attention(hidden_size, nhead, dropout, ctx_dim=cross_size1)
        self.cross_attention_2 = Attention(hidden_size, nhead, dropout, ctx_dim=cross_size2)
        self.self_attention = Attention(hidden_size, nhead, dropout)

        self.out1 = AttentionOutput(hidden_size, dropout)
        self.out2 = AttentionOutput(hidden_size, dropout)
        self.out3 = AttentionOutput(hidden_size, dropout)
        self.out4 = AttentionOutput(hidden_size, dropout)
        self.attention = BiAttention(hidden_size,hidden_size,nhead,dropout)
        self.gate = Gate(hidden_size, hidden_size)

    def forward(self, hidden_states, cross_states_1, cross_states_2, origin_states, attention_mask=None):
        hidden_states = self.attention(hidden_states, origin_states, attention_mask=attention_mask)

        cross_1 = self.cross_attention_1(hidden_states, cross_states_1, attention_mask=attention_mask)
        cross_1 = self.out1(cross_1, hidden_states)

        cross_2 = self.cross_attention_2(hidden_states, cross_states_2, attention_mask=attention_mask)
        cross_2 = self.out2(cross_2, hidden_states)

        cross_fusion = self.gate(cross_1, cross_2)
        cross_fusion = self.out4(cross_fusion, cross_fusion)

        self_out = self.self_attention(cross_fusion, cross_fusion, attention_mask=attention_mask)
        self_out = self.out3(self_out, cross_fusion)


        return self_out
