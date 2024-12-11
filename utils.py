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