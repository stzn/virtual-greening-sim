"""実データ（RR 間隔の CSV）の読み込み口の振る舞い。"""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from vmgs.analysis.io import RecordingFormatError, read_recordings, write_recordings
from vmgs.analysis.synthetic import Design, simulate_study


def test_合成データを書き出して読み込むと同じになる(tmp_path: Path) -> None:
    study = simulate_study(Design(n_subjects=2, repeats=1), seed=0)
    beats, sessions = tmp_path / "beats.csv", tmp_path / "sessions.csv"
    write_recordings(study.recordings, beats, sessions, origin=datetime(2026, 7, 1, tzinfo=UTC))

    loaded = read_recordings(beats, sessions)

    assert [r.session for r in loaded] == [r.session for r in study.recordings]
    for a, b in zip(loaded, study.recordings, strict=True):
        assert a.subject == b.subject
        assert a.condition == b.condition
        assert (a.walk, a.leave) == (b.walk, b.leave)
        assert a.pet_c == pytest.approx(b.pet_c)
        assert np.allclose(a.rr_ms, b.rr_ms, atol=1e-3)
        assert np.allclose(a.beat_times_s, b.beat_times_s, atol=1e-3)


def test_秒の数値の時刻も読める(tmp_path: Path) -> None:
    beats = tmp_path / "beats.csv"
    sessions = tmp_path / "sessions.csv"
    beats.write_text(
        "subject,session,timestamp,rr_ms\nP01,s1,100.8,800\nP01,s1,101.6,800\nP01,s1,102.4,800\n",
        encoding="utf-8",
    )
    sessions.write_text(
        "session,subject,condition,contact_start,green,pet_c,noise_db\n"
        "s1,P01,sit-leave,101.6,0.7,30.5,52\n",
        encoding="utf-8",
    )
    (rec,) = read_recordings(beats, sessions)
    # 接触開始からの秒にそろえる
    assert np.allclose(rec.beat_times_s, [-0.8, 0.0, 0.8])
    assert rec.leave and not rec.walk


def test_必須の列がないと列名を挙げて失敗する(tmp_path: Path) -> None:
    beats = tmp_path / "beats.csv"
    sessions = tmp_path / "sessions.csv"
    beats.write_text("subject,session,time,rr\nP01,s1,1,800\n", encoding="utf-8")
    sessions.write_text(
        "session,subject,condition,contact_start,green,pet_c,noise_db\n"
        "s1,P01,sit-leave,0,0.7,30,52\n",
        encoding="utf-8",
    )
    with pytest.raises(RecordingFormatError, match="rr_ms"):
        read_recordings(beats, sessions)


def test_知らない条件名は失敗する(tmp_path: Path) -> None:
    beats = tmp_path / "beats.csv"
    sessions = tmp_path / "sessions.csv"
    beats.write_text("subject,session,timestamp,rr_ms\nP01,s1,1,800\n", encoding="utf-8")
    sessions.write_text(
        "session,subject,condition,contact_start,green,pet_c,noise_db\ns1,P01,park,0,0.7,30,52\n",
        encoding="utf-8",
    )
    with pytest.raises(RecordingFormatError, match="park"):
        read_recordings(beats, sessions)
