"""The MCP surface as a client sees it, driven in-process through the SDK's Client. No device needed: only the tools
that never touch it are called."""
import json

import anyio
import pytest
from mcp import Client

from ko_bridge import knowledge, server as srv
from ko_bridge import config as C


def _data(result):
    return result.structured_content if result.structured_content is not None else json.loads(result.content[0].text)


def test_tool_list_is_typed_and_cached():
    async def go():
        async with Client(srv._server) as c:
            res = await c.list_tools()
            tools = {t.name: t for t in res.tools}
            expected = {"get_status", "get_guide", "human_steps", "set_config", "list_sounds", "describe_sound", "audition_sound",
                        "upload_sound", "upload_sounds", "delete_sound", "describe_project", "write_project", "assign_pads",
                        "write_pattern", "select_project", "list_checkpoints", "restore_checkpoint", "backup_device",
                        "play_pattern", "play_notes", "record_audio", "list_takes", "get_take", "measure_take",
                        "job_status", "list_jobs", "cancel_job"}
            assert expected <= set(tools), sorted(expected - set(tools))
            assert res.ttl_ms == 3_600_000
            assert all(t.title for t in res.tools)
            schema = tools["write_project"].input_schema
            spec = schema["$defs"]["ProjectSpec"]["properties"]
            assert {"bpm", "pads", "patterns", "scene", "fader_bases", "fx"} <= set(spec)
            pad = schema["$defs"]["PadSpec"]["properties"]
            assert pad["pad"]["enum"] == list(knowledge.__dict__.get("PAD_LABELS", (".", "0", "enter", "1", "2", "3", "4", "5", "6", "7", "8", "9")))
            assert (schema["properties"]["project"]["minimum"], schema["properties"]["project"]["maximum"]) == (1, 9)
            assert set(tools["human_steps"].input_schema["properties"]["topic"]["enum"]) == set(knowledge.HUMAN_STEPS)
            assert tools["delete_sound"].annotations.destructive_hint is True
            assert tools["describe_project"].annotations.read_only_hint is True
    anyio.run(go)


def test_guides_and_human_steps_over_mcp():
    async def go():
        async with Client(srv._server) as c:
            r = await c.call_tool("get_guide", {"topic": "quickstart"})
            assert "list_sounds" in r.content[0].text and "write_project" in r.content[0].text
            r = await c.call_tool("human_steps", {"topic": "midi_clock_in"})
            assert "SHIFT+ERASE" in r.content[0].text and "1 0 1" in r.content[0].text
            r = await c.call_tool("describe_project", {"project": 10})   # the schema refuses 10 before any device I/O
            assert r.is_error and "9" in r.content[0].text
    anyio.run(go)


def test_set_config_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "HOME", str(tmp_path))
    out = srv.set_config(midi_clock="in", default_bpm=96.0, scratch_project=9)
    assert out == {"midi_clock": "in", "default_bpm": 96.0, "scratch_project": 9}
    assert C.Config.load().default_bpm == 96.0
    assert srv.get_take.__name__ == "get_take"
    with pytest.raises(Exception):
        srv.get_take("../config")


def test_project_spec_compiles_over_a_saved_archive():
    """Compiling a spec over a device archive from the spike backup needs no device."""
    import glob
    from ko_bridge import project as PJ
    tars = sorted(glob.glob("/Users/tyler/Music/ko-bridge/backups/*/projects/P02.tar"))
    if not tars:
        pytest.skip("no backup archive to compile over")
    base = open(tars[-1], "rb").read()
    spec = PJ.ProjectSpec(bpm=100, pads=[PJ.PadSpec(group="C", pad=".", slot=600, trim_end=93750)],
                          patterns=[PJ.PatternSpec(group="C", number=2, bars=1, events=[PJ.NoteEvent(pad=".", step=0), PJ.NoteEvent(pad=".", step=8, velocity=80)],
                                                   lanes=[PJ.Lane(parameter="lowpass", points=[PJ.LanePoint(step=0, value=1.0), PJ.LanePoint(step=8, value=0.4)])])])
    tar, doc = PJ.compile_spec(spec, base, {})
    desc = PJ.describe(tar)
    assert desc["bpm"] == 100.0
    assert desc["pads"]["C:."]["slot"] == 600 and desc["pads"]["C:."]["root_note"] == 60
    c02 = desc["patterns"]["C02"]
    assert [e["step"] for e in c02["events"]] == [0, 8] and c02["events"][0]["note"] == 60
    assert c02["lanes"]["lowpass"][1]["value"] == pytest.approx(0.4, abs=1e-3)
    assert desc["scenes"][0]["patterns"] == {"A": 1, "B": 1, "C": 2, "D": 1} and desc["scene_cursor"] == 1
    assert "B01" in desc["patterns"] and desc["patterns"]["B01"]["events"] == []  # silent groups get empty members
