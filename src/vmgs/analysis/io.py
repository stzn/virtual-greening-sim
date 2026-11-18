"""実データ（RR 間隔の CSV）の読み込み口。形式は docs/analysis-data-format.md（仮決め）。

- 拍の CSV: subject, session, timestamp, rr_ms（1 行 1 拍）
- セッションの CSV: session, subject, condition, contact_start, green, pet_c, noise_db
  （追跡があれば、省略できる列 contact_end, track_start, track_end を 3 つそろえて足す）
- 出来事の CSV（省略可）: session, time, event（event は talk・phone・food・away）
- 環境の CSV（省略可）: session, time, ta_c, rh_pct, wind_ms, globe_c, laeq_db

timestamp・contact_start は ISO 8601 の日時（タイムゾーン付きを推奨）か、秒の数値。
2 つのファイルで同じ表し方にする。
"""

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vmgs.analysis.recording import CONDITIONS, Recording

BEAT_COLUMNS = ("subject", "session", "timestamp", "rr_ms")
SESSION_COLUMNS = ("session", "subject", "condition", "contact_start", "green", "pet_c", "noise_db")
TRACK_COLUMNS = ("contact_end", "track_start", "track_end")
EVENT_COLUMNS = ("session", "time", "event")
ENVIRONMENT_COLUMNS = ("session", "time", "ta_c", "rh_pct", "wind_ms", "globe_c", "laeq_db")


class RecordingFormatError(ValueError):
    """CSV の形式が仮決めの形式と合わない。"""


def _require(df: pd.DataFrame, columns: Sequence[str], path: Path) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise RecordingFormatError(f"{path}: 必須の列がない: {', '.join(missing)}")


def _to_seconds(values: pd.Series) -> pd.Series:
    """秒の数値はそのまま、日時の文字列は UNIX 時刻の秒にする。"""
    if pd.api.types.is_numeric_dtype(values):
        return values.astype(float)
    dt = pd.to_datetime(values, utc=True, format="ISO8601")
    return (dt - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()


def _relative(value: Any, t0: float) -> float | None:
    """接触開始からの秒。空欄は None。"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return float(value) - t0


def _read_events(path: Path, sessions: pd.Series) -> pd.DataFrame:
    events = pd.read_csv(path, dtype={"session": str, "event": str})
    _require(events, EVENT_COLUMNS, path)
    unknown = sorted(set(events["session"]) - set(sessions))
    if unknown:
        raise RecordingFormatError(f"{path}: セッションの CSV に無い session: {', '.join(unknown)}")
    return events.assign(t=_to_seconds(events["time"]))


def read_recordings(
    beats_path: Path, sessions_path: Path, events_path: Path | None = None
) -> list[Recording]:
    """CSV を読み、セッションの CSV の順に Recording を返す（時刻は接触開始が 0 秒）。

    セッションの CSV に追跡の列があれば追跡の時刻を、events_path を渡せば出来事を読む。
    """
    beats = pd.read_csv(beats_path, dtype={"subject": str, "session": str})
    sessions = pd.read_csv(sessions_path, dtype={"subject": str, "session": str})
    _require(beats, BEAT_COLUMNS, beats_path)
    _require(sessions, SESSION_COLUMNS, sessions_path)
    unknown = sorted(set(sessions["condition"]) - set(CONDITIONS))
    if unknown:
        raise RecordingFormatError(
            f"{sessions_path}: 知らない条件: {', '.join(map(str, unknown))}"
            f"（{', '.join(CONDITIONS)} のどれか）"
        )
    if sessions["session"].duplicated().any():
        raise RecordingFormatError(f"{sessions_path}: session が重複している")
    beats = beats.assign(t=_to_seconds(beats["timestamp"]))
    sessions = sessions.assign(t0=_to_seconds(sessions["contact_start"]))
    has_track = any(c in sessions.columns for c in TRACK_COLUMNS)
    if has_track:
        _require(sessions, TRACK_COLUMNS, sessions_path)
        for c in TRACK_COLUMNS:
            filled = sessions[c].notna()
            sessions[f"{c}_t"] = np.nan
            if filled.any():
                sessions.loc[filled, f"{c}_t"] = _to_seconds(sessions.loc[filled, c]).to_numpy()
    events = (
        _read_events(events_path, sessions["session"])
        if events_path is not None
        else pd.DataFrame(columns=["session", "t", "event"])
    )

    out: list[Recording] = []
    for row in sessions.to_dict("records"):
        b = beats[(beats["session"] == row["session"]) & (beats["subject"] == row["subject"])]
        b = b.sort_values("t")
        t0 = float(row["t0"])
        track = [_relative(row.get(f"{c}_t"), t0) if has_track else None for c in TRACK_COLUMNS]
        ev = events[events["session"] == row["session"]].sort_values("t")
        try:
            out.append(
                Recording(
                    session=str(row["session"]),
                    subject=str(row["subject"]),
                    condition=str(row["condition"]),
                    green=float(row["green"]),
                    pet_c=float(row["pet_c"]),
                    noise_db=float(row["noise_db"]),
                    beat_times_s=(b["t"].to_numpy(dtype=float) - t0),
                    rr_ms=b["rr_ms"].to_numpy(dtype=float),
                    contact_end_s=track[0],
                    track_start_s=track[1],
                    track_end_s=track[2],
                    events=tuple(
                        (float(t) - t0, str(e)) for t, e in zip(ev["t"], ev["event"], strict=True)
                    ),
                )
            )
        except ValueError as e:
            raise RecordingFormatError(f"{sessions_path}: session {row['session']}: {e}") from e
    return out


def read_environment(path: Path, contact_starts: Mapping[str, float]) -> pd.DataFrame:
    """環境の CSV を読み、時刻を各セッションの接触開始からの秒（列 t）にする。

    contact_starts はセッション → 接触開始（環境の CSV の time と同じ表し方の秒）。
    """
    env = pd.read_csv(path, dtype={"session": str})
    _require(env, ENVIRONMENT_COLUMNS, path)
    unknown = sorted(set(env["session"]) - set(contact_starts))
    if unknown:
        raise RecordingFormatError(f"{path}: 接触開始の分からない session: {', '.join(unknown)}")
    t = _to_seconds(env["time"]) - env["session"].map(contact_starts).astype(float)
    return env.assign(t=t)


def contact_environment(env: pd.DataFrame, session: str, contact_end_s: float) -> dict[str, float]:
    """接触中（0 秒以上 contact_end_s 未満）の環境の平均。PET・騒音をセッションの CSV に書く元。"""
    rows = env[(env["session"] == session) & (env["t"] >= 0) & (env["t"] < contact_end_s)]
    if rows.empty:
        raise ValueError(f"session {session} の接触中の環境の記録が無い")
    return {c: float(rows[c].mean()) for c in ENVIRONMENT_COLUMNS[2:]}


def _iso(start: pd.Timestamp, seconds: float) -> str:
    """接触開始 start から seconds 秒後の日時（ISO 8601）。"""
    return str((start + pd.Timedelta(seconds=seconds)).isoformat())


def write_recordings(
    recordings: Sequence[Recording],
    beats_path: Path,
    sessions_path: Path,
    origin: datetime,
    events_path: Path | None = None,
) -> None:
    """読み込みと同じ形式で書き出す（合成データの確認用）。セッションは 2 時間ずつずらす。

    追跡のある記録が 1 つでもあれば追跡の列を足し、events_path を渡せば出来事の CSV も書く。
    """
    beat_rows: list[pd.DataFrame] = []
    session_rows = []
    event_rows = []
    tracked = any(r.tracked for r in recordings)
    for i, r in enumerate(recordings):
        # 追跡（最大 60 分）の拍が次のセッションに重ならないよう 2 時間ずつずらす
        start = pd.Timestamp(origin + timedelta(hours=2 * i))
        stamps = start + pd.to_timedelta(np.round(r.beat_times_s * 1e6), unit="us")
        beat_rows.append(
            pd.DataFrame(
                {
                    "subject": r.subject,
                    "session": r.session,
                    "timestamp": stamps.map(lambda x: x.isoformat()),
                    "rr_ms": np.round(r.rr_ms, 4),
                }
            )
        )
        session_rows.append(
            {
                "session": r.session,
                "subject": r.subject,
                "condition": r.condition,
                "contact_start": start.isoformat(),
                "green": r.green,
                "pet_c": r.pet_c,
                "noise_db": r.noise_db,
                **(
                    {
                        c: (_iso(start, v) if v is not None else None)
                        for c, v in zip(
                            TRACK_COLUMNS,
                            (r.contact_end_s, r.track_start_s, r.track_end_s),
                            strict=True,
                        )
                    }
                    if tracked
                    else {}
                ),
            }
        )
        event_rows += [
            {"session": r.session, "time": _iso(start, t), "event": e} for t, e in r.events
        ]
    pd.concat(beat_rows).to_csv(beats_path, index=False)
    pd.DataFrame(session_rows).to_csv(sessions_path, index=False)
    if events_path is not None:
        pd.DataFrame(event_rows, columns=list(EVENT_COLUMNS)).to_csv(events_path, index=False)
