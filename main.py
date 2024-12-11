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



if __name__ == "__main__":
    main()