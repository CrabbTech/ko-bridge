"""The producer layer's surface without the device or the cloud: the secrets loader's rules, the Lyria response parsing
against an Interaction built from a real WAV, the labelled no-model paths, the request schemas google-genai accepts, the
trim default of a pad, and what a client sees when Lyria is unavailable. Nothing here spends money or touches the EP-133."""
import base64
import glob
import io
import json
import os

import anyio
import numpy as np
import pytest
import soundfile as sf
from mcp import Client

from ko_bridge import config as C
from ko_bridge import gemini_producer as GP
from ko_bridge import project as PJ
from ko_bridge import server as srv

genai = pytest.importorskip("google.genai", reason="the gemini extra is not installed (uv sync --extra gemini)")
from google.genai import interactions as I  # noqa: E402


@pytest.fixture
def no_key(monkeypatch):
    """The process without a Gemini key: the environment cleared and the secrets files not consulted."""
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(C, "_SECRETS_REPORT", {"file": None, "loaded": [], "already_set": [], "skipped_lines": 0})


def test_load_secrets_rules(tmp_path, monkeypatch):
    f = tmp_path / "secrets.env"
    f.write_text("# comment\n\nexport FOO_A=1\nBAR_B='quoted value'\nnoequals\n" + "apikey_" + "a" * 100 + "=blob\n"
                 "9BAD=1\nOK_C=\"dq\"\nPRESET=override\n")
    monkeypatch.setattr(C, "_SECRETS_REPORT", None)
    env = {"PRESET": "kept"}
    report = C.load_secrets((str(tmp_path / "missing.env"), str(f)), environ=env)
    assert env == {"PRESET": "kept", "FOO_A": "1", "BAR_B": "quoted value", "OK_C": "dq"}, "quotes stripped, export accepted, nothing overridden"
    assert report["file"] == str(f) and report["loaded"] == ["FOO_A", "BAR_B", "OK_C"] and report["already_set"] == ["PRESET"]
    assert report["skipped_lines"] == 3, "no '=', a 106-character name, a name starting with a digit"
    assert "blob" not in json.dumps(report) and "quoted" not in json.dumps(report), "the report carries names, never values"
    assert C.load_secrets((str(tmp_path / "other.env"),), environ={}) is report, "runs once per process unless forced"
    assert C.load_secrets((str(tmp_path / "other.env"),), environ={}, force=True)["file"] is None


def _wav_bytes(seconds=1.0, silence=0.2, sr=44100):
    t = np.arange(int(seconds * sr)) / sr
    x = np.zeros((len(t), 2), dtype=np.float32)
    x[int(silence * sr):, 0] = 0.5 * np.sin(2 * np.pi * 220 * t[int(silence * sr):])
    x[:, 1] = x[:, 0]
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue(), sr


def test_lyria_audio_block_and_trim():
    wav, sr = _wav_bytes()
    block = {"type": "audio", "data": base64.b64encode(wav).decode(), "mime_type": "audio/wav", "sample_rate": sr, "channels": 2}
    filled = I.Interaction.model_validate({"id": "a", "status": "completed", "model": GP.LYRIA_MODEL, "output_audio": block})
    steps_only = I.Interaction.model_validate({"id": "b", "status": "completed", "model": GP.LYRIA_MODEL,
                                               "steps": [{"type": "model_output", "content": [{"type": "text", "text": "verse"}, block]}]})
    for interaction in (filled, steps_only):
        audio = GP._audio_block(interaction)
        assert audio is not None and audio.mime_type == "audio/wav"
        data, rate = sf.read(io.BytesIO(GP._audio_bytes(audio)), dtype="float32", always_2d=True)
        assert rate == sr and data.shape == (sr, 2)
        loop, onset_s = GP._first_bars(data, rate, 0.5)
        assert abs(onset_s - 0.2) < 0.01 and abs(len(loop) / rate - 0.5) < 0.01, "the loop starts at the first onset and lasts the requested bars"
    assert GP._audio_block(I.Interaction.model_validate({"id": "c", "status": "completed", "model": GP.LYRIA_MODEL, "output_text": "no audio"})) is None
    with pytest.raises(GP.LyriaError):
        GP._audio_bytes(I.AudioContent.model_validate({"type": "audio", "mime_type": "audio/wav"}))


def test_without_a_key_lyria_raises_and_the_other_paths_say_no_model(no_key, tmp_path):
    with pytest.raises(GP.GeminiUnavailable, match="GEMINI_API_KEY"):
        GP.generate_lyria_stem("soul loop", bpm=90, bars=2)
    plan, model = GP.compile_beat_with_gemini("Houston screw", bpm=68, bars=2)
    assert model == GP.HEURISTIC_COMPILER and plan.bars == 2 and plan.drums
    wav, sr = _wav_bytes()
    take = tmp_path / "take.wav"
    take.write_bytes(wav)
    crit = GP.critique_take_with_gemini(str(take))
    assert crit["model"] == GP.DSP_CRITIC and crit["note"] == GP.NO_KEY_NOTE and 1 <= crit["critique"]["overall_score"] <= 100
    assert GP.critique_take_with_gemini(str(take), use_gemini=False)["note"] == GP.NO_MODEL_REQUESTED_NOTE
    assert GP.no_model_note(False) == GP.NO_MODEL_REQUESTED_NOTE


def test_error_text_never_carries_the_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaFAKEKEY1234567890")
    scrubbed = GP._scrub("400 for https://example/v1?key=AIzaFAKEKEY1234567890 with header AIzaFAKEKEY1234567890", api_key="explicit-key-value")
    assert "AIzaFAKEKEY1234567890" not in scrubbed and scrubbed.count("<GEMINI_API_KEY>") == 2
    assert GP._scrub("explicit-key-value in a message", api_key="explicit-key-value") == "<GEMINI_API_KEY> in a message"


def test_request_schemas_are_accepted_by_the_sdk():
    """generate_content converts the Pydantic schema before any request; lt=64 (exclusiveMaximum) once failed here."""
    from google.genai import _transformers as T
    client = genai.Client(api_key="offline-schema-check")
    for model in (GP.GeminiBeatPlan, GP.GeminiAudioCritique):
        schema = T.t_schema(client._api_client, model)
        assert schema.properties
    assert "exclusiveMaximum" not in json.dumps(GP.GeminiBeatPlan.model_json_schema())


def test_lyria_tool_reports_the_missing_key_over_mcp(no_key):
    async def go():
        async with Client(srv._server) as c:
            tools = {t.name: t for t in (await c.list_tools()).tools}
            assert tools["generate_lyria_chop_group"].annotations.read_only_hint is False
            assert "use_gemini" in tools["compile_beat_with_gemini"].input_schema["properties"]
            assert "use_gemini" in tools["critique_take"].input_schema["properties"]
            r = await c.call_tool("generate_lyria_chop_group", {"prompt": "soul loop", "bpm": 90, "bars": 2, "upload": False})
            assert r.is_error and "GEMINI_API_KEY" in r.content[0].text and "synth" not in r.content[0].text.lower()
            r = await c.call_tool("compile_beat_with_gemini", {"prompt": "Houston screw", "bpm": 68, "bars": 2})
            data = r.structured_content if r.structured_content is not None else json.loads(r.content[0].text)
            assert data["model"] == GP.HEURISTIC_COMPILER and data["note"] == GP.NO_KEY_NOTE
    anyio.run(go)


def test_pad_trim_default_is_the_rest_of_the_sample():
    tars = sorted(glob.glob(os.path.join(C.HOME, "backups", "*", "projects", "P02.tar")))
    if not tars:
        pytest.skip("no backup archive to compile over")
    base = open(tars[-1], "rb").read()

    def pad(**kw):
        tar, _doc = PJ.compile_spec(PJ.ProjectSpec(pads=[PJ.PadSpec(group="C", pad=".", slot=600, **kw)]), base, {600: 93750})
        return PJ.describe(tar)["pads"]["C:."]

    assert pad(trim_start=1000)["trim_end"] == 92750, "trim_end is a length: the rest of the sample, not the whole sample"
    assert pad(trim_start=93000, trim_end=750)["trim_end"] == 750, "start + length may reach the end exactly"
    with pytest.raises(ValueError, match="beyond the end"):
        pad(trim_start=93750)
    with pytest.raises(ValueError, match="exceeds"):
        pad(trim_start=93000, trim_end=751)
