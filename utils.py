import os
import pandas as pd
import re


def read_csv(data_dir, file_name):
    file_path = os.path.join(data_dir, file_name)
    try:
        return pd.read_csv(file_path)
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding='utf-8')


def store_data(data_obj: list, store_file: str, results_dir):
    full_path = os.path.join(results_dir, store_file)
    df = pd.DataFrame(data_obj)
    df.to_csv(full_path, index=False, encoding='utf-8')
    print(f"Data stored at: {full_path}")


def safe_parse_int(text, default=0):
    cleaned_text = re.sub(r'[^\d]', '', text)
    return int(cleaned_text) if cleaned_text.isdigit() else default


def multitask_eval_fn(multitask_model, model_name, features_dict, batch_size=8):
    preds_dict = {}
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    if torch.cuda.is_available():    
        device = torch.device("cuda")
#     for task_name in ["hyperbole", "metaphor","irony","sarcasm"]:
    for task_name in ["hyperbole", "metaphor"]:
        true_list=[]
        pred_list=[]
        val_len = len(features_dict[task_name]["validation"])
        acc = 0.0
        for index in range(0, val_len, batch_size):

            batch = features_dict[task_name]["validation"][
                index : min(index + batch_size, val_len)
            ]["doc"]
            labels = features_dict[task_name]["validation"][
                index : min(index + batch_size, val_len)
            ]["target"]
            inputs = tokenizer(batch, max_length=512, padding=True)
            inputs["input_ids"] = torch.LongTensor(inputs["input_ids"])
            inputs["attention_mask"] = torch.LongTensor(inputs["attention_mask"])
             inputs["token_type_ids"] = torch.LongTensor(inputs["token_type_ids"])
             print(type(inputs["input_ids"]))
             print(type(inputs["attention_mask"]))
             print(type(inputs["token_type_ids"]))
             print(inputs["input_ids"])
             print(inputs["attention_mask"])
            logits = multitask_model(task_name, **inputs.to(device))[0]

            predictions = torch.argmax(
                torch.FloatTensor(torch.softmax(logits, dim=1).detach().cpu().tolist()),
                dim=1,
            )
            true_list.extend(list(np.array(labels)))
            pred_list.extend(list(np.array(predictions)))
            acc += sum(np.array(predictions) == np.array(labels))
        acc = acc / val_len
        print(f"Task name: {task_name}")
        
        print("---------------------------------Confusion Matrix------------------------------------")
        final_create_confusion_matrix = confusion_matrix(true_list, pred_list)
        final_confusion_matrix_df = pd.DataFrame(final_create_confusion_matrix)
        print(final_confusion_matrix_df)
        # Precision, Recall and F1 score calculation
        final_eval_metrics = classification_report(true_list, pred_list, output_dict=True)
        print(final_eval_metrics)
        
        if task_name=="hyperbole":
            with open("../../results/hyperbole.json", "r") as file:
                data = json.load(file)
            data.append(final_eval_metrics)
            with open("../../results/hyperbole.json", "w") as file:
                json.dump(data, file, indent=4)
            
        elif task_name=="metaphor":
            with open("../../results/metaphor.json", "r") as file:
                data = json.load(file)
            data.append(final_eval_metrics)
            with open("../../results/metaphor.json", "w") as file:
                json.dump(data, file, indent=4)
        
        print("---------------------------------Evaluation Metrics------------------------------------")
        final_eval_metrics_df = pd.DataFrame(final_eval_metrics).transpose()
        final_eval_metrics_df = final_eval_metrics_df.iloc[: , :-1]
        print(final_eval_metrics_df)
