from types import SimpleNamespace

import numpy as np

from lmms_eval.api.task import ConfigurableMessagesTask


class AudioDecoder:
    def get_all_samples(self):
        return SimpleNamespace(samples=np.zeros(160, dtype=np.float32), sample_rate=16000)


def _build_task(visual):
    task = ConfigurableMessagesTask.__new__(ConfigurableMessagesTask)
    task._config = SimpleNamespace(
        doc_to_messages=None,
        doc_to_visual=lambda doc: [visual],
        doc_to_text=lambda doc: "transcribe this",
    )
    task.lmms_eval_specific_kwargs = None
    return task


def test_auto_doc_to_messages_preserves_audio_decoder_visuals():
    task = _build_task(AudioDecoder())

    messages = task.doc_to_messages({})[0]["content"]

    assert messages[0]["type"] == "audio"
    assert messages[0]["url"].__class__.__name__ == "AudioDecoder"
    assert messages[1] == {"type": "text", "text": "transcribe this"}


def test_auto_doc_to_messages_treats_audio_dict_as_audio():
    audio = {"array": np.zeros(160, dtype=np.float32), "sampling_rate": 16000}
    task = _build_task(audio)

    messages = task.doc_to_messages({})[0]["content"]

    assert messages[0] == {"type": "audio", "url": audio}
