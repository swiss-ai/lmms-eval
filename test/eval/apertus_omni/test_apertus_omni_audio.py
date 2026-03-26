import unittest
from types import SimpleNamespace
import sys
import types

import numpy as np
from PIL import Image

if "evaluate" not in sys.modules:
    sys.modules["evaluate"] = types.SimpleNamespace(load=lambda name: None)
if "sqlitedict" not in sys.modules:
    sys.modules["sqlitedict"] = types.SimpleNamespace(SqliteDict=dict)

from lmms_eval.models.chat.apertus_omni import ApertusOmniChat


class _FakeTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        return "formatted prompt"


class ApertusOmniAudioTest(unittest.TestCase):
    def _build_chat_model(self) -> ApertusOmniChat:
        model = ApertusOmniChat.__new__(ApertusOmniChat)
        model.skip_text_only = False
        model.skip_multi_image = False
        model.vq_hub = "BAAI/Emu3.5-VisionTokenizer"
        model.vision_tokenizer_device = "cuda:0"
        model.vision_tokenizer_dtype = "bfloat16"
        model.emu_min_pixels = 256 * 256
        model.emu_max_pixels = 1400 * 1400
        model.vq_trust_remote_code = True
        model.image_placeholder = "<|image|>"
        model.audio_tokenizer_type = "wavtokenizer"
        model.audio_tokenizer_name = "WavTokenizer40"
        model.audio_tokenizer_device = "cuda"
        model.audio_tokenizer_compile = False
        model.audio_target_sampling_rate = 24000
        model.audio_default_sampling_rate = 16000
        model.audio_token_offset = 262344
        model.audio_vocab_size = 4096
        model.audio_tokenizer_path = None
        model.tokenizer = _FakeTokenizer()
        model.task_dict = {}
        model._rank = 0
        model._world_size = 1
        model.debug_samples = 0
        model.debug_max_chars = 0
        model._debug_logged_samples = 0
        return model

    def test_normalize_decoded_audio_dict(self):
        model = self._build_chat_model()
        audios = model._normalize_audios(
            {"array": np.array([0.1, -0.2, 0.3], dtype=np.float32), "sampling_rate": 22050}
        )

        self.assertEqual(len(audios), 1)
        waveform, sr = audios[0]
        self.assertEqual(sr, 22050)
        self.assertEqual(waveform.dtype, np.float32)
        self.assertTrue(np.allclose(waveform, np.array([0.1, -0.2, 0.3], dtype=np.float32)))

    def test_make_one_request_packs_audio_and_audio_processor_kwargs(self):
        model = self._build_chat_model()
        sample = {
            "audio": {"array": np.array([0.1, 0.2, -0.3], dtype=np.float32), "sampling_rate": 16000},
        }
        model.task_dict = {"task": {"split": [sample]}}

        raw_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "url": sample["audio"]},
                    {"type": "text", "text": "Transcribe this clip."},
                ],
            }
        ]
        request = SimpleNamespace(arguments=("fallback context", lambda doc: raw_messages, {}, 0, "task", "split"))

        prompt_dict, _gen_kwargs, counters = model.make_one_request(request)

        self.assertEqual(counters["audio_present"], 1)
        self.assertEqual(counters["text_only"], 0)
        self.assertEqual(prompt_dict["prompt"], "formatted prompt")
        self.assertIn("audio", prompt_dict["multi_modal_data"])
        self.assertNotIn("image", prompt_dict["multi_modal_data"])
        audio_waveform, audio_sr = prompt_dict["multi_modal_data"]["audio"][0]
        self.assertEqual(audio_sr, 16000)
        self.assertEqual(audio_waveform.dtype, np.float32)

        mm_kwargs = prompt_dict["mm_processor_kwargs"]
        self.assertEqual(mm_kwargs["apertus_audio_tokenizer_type"], "wavtokenizer")
        self.assertEqual(mm_kwargs["apertus_audio_tokenizer_name"], "WavTokenizer40")
        self.assertEqual(mm_kwargs["apertus_audio_target_sampling_rate"], 24000)
        self.assertEqual(mm_kwargs["apertus_audio_tokenizer_device"], "cuda")
        self.assertFalse(mm_kwargs["apertus_audio_tokenizer_compile"])
        self.assertEqual(mm_kwargs["apertus_audio_token_offset"], 262344)
        self.assertEqual(mm_kwargs["apertus_audio_vocab_size"], 4096)

        tokenizer_messages = model.tokenizer.calls[0]["messages"]
        user_parts = tokenizer_messages[0]["content"]["parts"]
        self.assertEqual([part["type"] for part in user_parts], ["text"])

    def test_make_one_request_packs_image_and_audio_together(self):
        model = self._build_chat_model()
        sample = {
            "audio": {"array": np.array([0.0, 0.4, -0.1], dtype=np.float32), "sampling_rate": 16000},
            "image": Image.new("RGB", (2, 2), color="white"),
        }
        model.task_dict = {"task": {"split": [sample]}}

        raw_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": sample["image"]},
                    {"type": "audio", "url": sample["audio"]},
                    {"type": "text", "text": "Describe both modalities."},
                ],
            }
        ]
        request = SimpleNamespace(arguments=("fallback context", lambda doc: raw_messages, {}, 0, "task", "split"))

        prompt_dict, _gen_kwargs, counters = model.make_one_request(request)

        self.assertEqual(counters["audio_present"], 1)
        self.assertIn("image", prompt_dict["multi_modal_data"])
        self.assertIn("audio", prompt_dict["multi_modal_data"])
        self.assertEqual(len(prompt_dict["multi_modal_data"]["image"]), 1)
        self.assertEqual(len(prompt_dict["multi_modal_data"]["audio"]), 1)

        tokenizer_messages = model.tokenizer.calls[0]["messages"]
        user_parts = tokenizer_messages[0]["content"]["parts"]
        self.assertEqual([part["type"] for part in user_parts], ["image", "text"])

    def test_make_one_request_still_skips_video(self):
        model = self._build_chat_model()
        model.task_dict = {"task": {"split": [{}]}}
        raw_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "url": "/tmp/example.mp4"},
                    {"type": "text", "text": "Describe the video."},
                ],
            }
        ]
        request = SimpleNamespace(arguments=("fallback context", lambda doc: raw_messages, {}, 0, "task", "split"))

        prompt_dict, _gen_kwargs, counters = model.make_one_request(request)

        self.assertIsNone(prompt_dict)
        self.assertEqual(counters["unsupported_video"], 1)
        self.assertEqual(counters["skipped"], 1)


if __name__ == "__main__":
    unittest.main()
