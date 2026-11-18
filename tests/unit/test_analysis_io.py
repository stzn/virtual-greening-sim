"""実データ（RR 間隔の CSV）の読み込み口の振る舞い。"""

import dataclasses
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from vmgs.analysis.io import (
    RecordingFormatError,
    contact_environment,
    read_environment,
    read_recordings,
    write_recordings,
)
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


# --- 追跡の区間・出来事・環境（pilot-protocol.md 7 章の拡張） ---

_SESSION_HEAD = "session,subject,condition,contact_start,green,pet_c,noise_db"


def _write(tmp_path: Path, sessions: str, events: str | None = None) -> tuple[Path, Path, Path]:
    beats, sess, ev = tmp_path / "beats.csv", tmp_path / "sessions.csv", tmp_path / "events.csv"
    beats.write_text("subject,session,timestamp,rr_ms\nP01,s1,1,800\n", encoding="utf-8")
    sess.write_text(sessions, encoding="utf-8")
    if events is not None:
        ev.write_text(events, encoding="utf-8")
    return beats, sess, ev


def test_追跡の時刻と出来事は_接触開始からの秒で読む(tmp_path: Path) -> None:
    beats, sess, ev = _write(
        tmp_path,
        f"{_SESSION_HEAD},contact_end,track_start,track_end\n"
        "s1,P01,sit-leave,100,0.7,30,52,700,1000,4300\n",
        "session,time,event\ns1,1600,phone\ns1,2200,talk\n",
    )
    (rec,) = read_recordings(beats, sess, events_path=ev)
    assert (rec.contact_end_s, rec.track_start_s, rec.track_end_s) == (600.0, 900.0, 4200.0)
    assert rec.events == ((1500.0, "phone"), (2100.0, "talk"))


def test_追跡の列が無い今の形式も読め_追跡の時刻は無しになる(tmp_path: Path) -> None:
    beats, sess, _ = _write(tmp_path, f"{_SESSION_HEAD}\ns1,P01,sit-leave,0,0.7,30,52\n")
    (rec,) = read_recordings(beats, sess)
    assert rec.contact_end_s is None and rec.track_start_s is None and rec.track_end_s is None
    assert rec.events == ()


@pytest.mark.parametrize(
    ("sessions", "events", "reason"),
    [
        # 追跡の時刻の順が逆（接触の終わり ≦ 追跡の始め ≦ 追跡の終わり）
        (
            f"{_SESSION_HEAD},contact_end,track_start,track_end\ns1,P01,sit-leave,0,0.7,30,52,600,500,4000\n",
            None,
            "順",
        ),
        # 3 つの時刻の一部だけ
        (
            f"{_SESSION_HEAD},contact_end,track_start,track_end\ns1,P01,sit-leave,0,0.7,30,52,600,,4000\n",
            None,
            "そろって",
        ),
        (
            f"{_SESSION_HEAD}\ns1,P01,sit-leave,0,0.7,30,52\n",
            "session,time,event\ns1,10,sing\n",
            "sing",
        ),
        (
            f"{_SESSION_HEAD}\ns1,P01,sit-leave,0,0.7,30,52\n",
            "session,time,event\ns9,10,talk\n",
            "s9",
        ),
    ],
)
def test_追跡と出来事の形の誤りは理由つきで止める(
    tmp_path: Path, sessions: str, events: str | None, reason: str
) -> None:
    beats, sess, ev = _write(tmp_path, sessions, events)
    with pytest.raises(RecordingFormatError, match=reason):
        read_recordings(beats, sess, events_path=ev if events is not None else None)


def test_追跡ありの合成データを書き出して読み込むと_追跡の時刻と出来事も同じになる(
    tmp_path: Path,
) -> None:
    study = simulate_study(Design(n_subjects=1, repeats=1, track_s=900.0), seed=0)
    recs = [
        dataclasses.replace(study.recordings[0], events=((1000.0, "phone"),)),
        *study.recordings[1:],
    ]
    paths = tmp_path / "beats.csv", tmp_path / "sessions.csv", tmp_path / "events.csv"
    write_recordings(
        recs, *paths[:2], origin=datetime(2026, 7, 1, tzinfo=UTC), events_path=paths[2]
    )
    loaded = read_recordings(*paths[:2], events_path=paths[2])
    for a, b in zip(loaded, recs, strict=True):
        assert a.contact_end_s == pytest.approx(b.contact_end_s)
        assert a.track_start_s == pytest.approx(b.track_start_s)
        assert a.track_end_s == pytest.approx(b.track_end_s)
        assert [e for _, e in a.events] == [e for _, e in b.events]
        assert [t for t, _ in a.events] == pytest.approx([t for t, _ in b.events])


def test_環境の記録を読み_接触中の平均を出す(tmp_path: Path) -> None:
    env = tmp_path / "environment.csv"
    env.write_text(
        "session,time,ta_c,rh_pct,wind_ms,globe_c,laeq_db\n"
        "s1,-60,26,60,0.5,27,50\ns1,0,30,55,1.0,34,58\ns1,300,32,50,2.0,36,62\ns1,700,27,58,0.4,28,51\n",
        encoding="utf-8",
    )
    starts = {"s1": 0.0}
    table = read_environment(env, starts)
    mean = contact_environment(table, "s1", contact_end_s=600.0)
    # 接触中（0〜600 秒）の 2 行だけの平均
    assert mean["ta_c"] == pytest.approx(31.0)
    assert mean["laeq_db"] == pytest.approx(60.0)
