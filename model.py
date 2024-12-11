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