import os
from collections import OrderedDict

from lmms_eval.tasks.asr_wer_utils import (
    EvaluationTokenizer as SharedEvaluationTokenizer,
)
from lmms_eval.tasks.asr_wer_utils import compute_wer as shared_compute_wer
from lmms_eval.tasks.asr_wer_utils import remove_sp as shared_remove_sp
from lmms_eval.tasks.asr_wer_utils import (
    zhconv_convert,
)
from lmms_eval.tasks.librispeech.cn_tn import TextNorm
from lmms_eval.tasks.librispeech.whisper_normalizer.basic import BasicTextNormalizer
from lmms_eval.tasks.librispeech.whisper_normalizer.english import EnglishTextNormalizer

_FLEURS_LANG_TO_ID = OrderedDict(
    [
        ("Afrikaans", "af"),
        ("Amharic", "am"),
        ("Arabic", "ar"),
        ("Armenian", "hy"),
        ("Assamese", "as"),
        ("Asturian", "ast"),
        ("Azerbaijani", "az"),
        ("Belarusian", "be"),
        ("Bengali", "bn"),
        ("Bosnian", "bs"),
        ("Bulgarian", "bg"),
        ("Burmese", "my"),
        ("Catalan", "ca"),
        ("Cebuano", "ceb"),
        ("Mandarin Chinese", "cmn_hans"),
        ("Cantonese Chinese", "yue_hant"),
        ("Croatian", "hr"),
        ("Czech", "cs"),
        ("Danish", "da"),
        ("Dutch", "nl"),
        ("English", "en"),
        ("Estonian", "et"),
        ("Filipino", "fil"),
        ("Finnish", "fi"),
        ("French", "fr"),
        ("Fula", "ff"),
        ("Galician", "gl"),
        ("Ganda", "lg"),
        ("Georgian", "ka"),
        ("German", "de"),
        ("Greek", "el"),
        ("Gujarati", "gu"),
        ("Hausa", "ha"),
        ("Hebrew", "he"),
        ("Hindi", "hi"),
        ("Hungarian", "hu"),
        ("Icelandic", "is"),
        ("Igbo", "ig"),
        ("Indonesian", "id"),
        ("Irish", "ga"),
        ("Italian", "it"),
        ("Japanese", "ja"),
        ("Javanese", "jv"),
        ("Kabuverdianu", "kea"),
        ("Kamba", "kam"),
        ("Kannada", "kn"),
        ("Kazakh", "kk"),
        ("Khmer", "km"),
        ("Korean", "ko"),
        ("Kyrgyz", "ky"),
        ("Lao", "lo"),
        ("Latvian", "lv"),
        ("Lingala", "ln"),
        ("Lithuanian", "lt"),
        ("Luo", "luo"),
        ("Luxembourgish", "lb"),
        ("Macedonian", "mk"),
        ("Malay", "ms"),
        ("Malayalam", "ml"),
        ("Maltese", "mt"),
        ("Maori", "mi"),
        ("Marathi", "mr"),
        ("Mongolian", "mn"),
        ("Nepali", "ne"),
        ("Northern-Sotho", "nso"),
        ("Norwegian", "nb"),
        ("Nyanja", "ny"),
        ("Occitan", "oc"),
        ("Oriya", "or"),
        ("Oromo", "om"),
        ("Pashto", "ps"),
        ("Persian", "fa"),
        ("Polish", "pl"),
        ("Portuguese", "pt"),
        ("Punjabi", "pa"),
        ("Romanian", "ro"),
        ("Russian", "ru"),
        ("Serbian", "sr"),
        ("Shona", "sn"),
        ("Sindhi", "sd"),
        ("Slovak", "sk"),
        ("Slovenian", "sl"),
        ("Somali", "so"),
        ("Sorani-Kurdish", "ckb"),
        ("Spanish", "es"),
        ("Swahili", "sw"),
        ("Swedish", "sv"),
        ("Tajik", "tg"),
        ("Tamil", "ta"),
        ("Telugu", "te"),
        ("Thai", "th"),
        ("Turkish", "tr"),
        ("Ukrainian", "uk"),
        ("Umbundu", "umb"),
        ("Urdu", "ur"),
        ("Uzbek", "uz"),
        ("Vietnamese", "vi"),
        ("Welsh", "cy"),
        ("Wolof", "wo"),
        ("Xhosa", "xh"),
        ("Yoruba", "yo"),
        ("Zulu", "zu"),
    ]
)
_FLEURS_LANG_SHORT_TO_LONG = {v: k for k, v in _FLEURS_LANG_TO_ID.items()}
CER_LANGUAGES = {"cmn_hans", "th", "yue_hant"}
PROMPT_LANGUAGE_NAMES = {
    "cmn_hans": "Mandarin Chinese using Simplified Chinese characters",
    "yue_hant": "Cantonese Chinese using Traditional Chinese characters",
}

# ImportError: To support decoding audio files, please install 'librosa' and 'soundfile'.
english_normalizer = EnglishTextNormalizer()
chinese_normalizer = TextNorm(
    to_banjiao=False,
    to_upper=False,
    to_lower=False,
    remove_fillers=False,
    remove_erhua=False,
    check_chars=False,
    remove_space=False,
    cc_mode="",
)
basic_normalizer = BasicTextNormalizer()

dir_name = os.path.dirname(os.path.abspath(__file__))


def safe_english_normalizer(text):
    try:
        return english_normalizer(text)
    except AssertionError:
        return basic_normalizer(text)


def fleurs_doc_to_audio(doc):
    return [doc["audio"]]


def fleurs_doc_to_text(doc, lmms_eval_specific_kwargs):
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    lan = _FLEURS_LANG_TO_ID[doc["language"]]
    prompt_language = PROMPT_LANGUAGE_NAMES.get(lan, doc["language"])
    prompt = "Transcribe the spoken audio exactly. " f"Output only the transcript in {prompt_language}. " "Do not translate, explain, repeat, or add any extra text:"
    return f"{pre_prompt}{prompt}{post_prompt}"


def fleurs_process_result(doc, result):
    pred = result[0] if len(result) > 0 else ""

    gt = doc["transcription"]
    source = doc["path"]
    language = doc["language"]
    lan = _FLEURS_LANG_TO_ID[language]

    data_dict = {"gt": gt, "pred": pred, "source": source, "language": language}

    metric = "cer" if lan in CER_LANGUAGES else "wer"
    return {metric: data_dict}


PUNCS = "!,.?;:"


def remove_sp(text, language):
    return shared_remove_sp(text, language, puncs=PUNCS, no_space_languages={"cmn_hans", "th"})


EvaluationTokenizer = SharedEvaluationTokenizer


def compute_wer(refs, hyps, language):
    return shared_compute_wer(
        refs,
        hyps,
        language,
        yue_languages={"yue_hant"},
        english_languages={"en"},
        chinese_languages={"cmn_hans"},
        char_languages=CER_LANGUAGES,
        yue_converter=zhconv_convert,
        english_normalizer=safe_english_normalizer,
        chinese_normalizer=chinese_normalizer,
        basic_normalizer=basic_normalizer,
    )


def _fleurs_error_rate(results):
    refs, hyps = [], []
    lan = ""
    for result in results:
        lan = _FLEURS_LANG_TO_ID[result["language"]]
        gt = result["gt"]
        response = result["pred"]
        gt = remove_sp(gt, lan)
        response = remove_sp(response, lan)
        refs.append(gt)
        hyps.append(response)
    return compute_wer(refs, hyps, lan) * 100, lan


def fleurs_wer(results, args):
    error_rate, lan = _fleurs_error_rate(results)
    if lan in CER_LANGUAGES:
        raise ValueError(f"Language {lan} is character-level; use fleurs_cer instead of fleurs_wer.")
    return error_rate


def fleurs_cer(results, args):
    error_rate, lan = _fleurs_error_rate(results)
    if lan not in CER_LANGUAGES:
        raise ValueError(f"Language {lan} is word-level; use fleurs_wer instead of fleurs_cer.")
    return error_rate
