"""実データ（RR 間隔の CSV）の読み込み口。形式は docs/analysis-data-format.md（仮決め）。

- 拍の CSV: subject, session, timestamp, rr_ms（1 行 1 拍）
- セッションの CSV: session, subject, condition, contact_start, green, pet_c, noise_db

timestamp・contact_start は ISO 8601 の日時（タイムゾーン付きを推奨）か、秒の数値。
2 つのファイルで同じ表し方にする。
"""

from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from vmgs.analysis.recording import CONDITIONS, Recording

BEAT_COLUMNS = ("subject", "session", "timestamp", "rr_ms")
SESSION_COLUMNS = ("session", "subject", "condition", "contact_start", "green", "pet_c", "noise_db")


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


def read_recordings(beats_path: Path, sessions_path: Path) -> list[Recording]:
    """2 つの CSV を読み、セッションの CSV の順に Recording を返す（時刻は接触開始が 0 秒）。"""
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

    out: list[Recording] = []
    for row in sessions.to_dict("records"):
        b = beats[(beats["session"] == row["session"]) & (beats["subject"] == row["subject"])]
        b = b.sort_values("t")
        out.append(
            Recording(
                session=str(row["session"]),
                subject=str(row["subject"]),
                condition=str(row["condition"]),
                green=float(row["green"]),
                pet_c=float(row["pet_c"]),
                noise_db=float(row["noise_db"]),
                beat_times_s=(b["t"].to_numpy(dtype=float) - float(row["t0"])),
                rr_ms=b["rr_ms"].to_numpy(dtype=float),
            )
        )
    return out


def write_recordings(
    recordings: Sequence[Recording], beats_path: Path, sessions_path: Path, origin: datetime
) -> None:
    """読み込みと同じ形式で書き出す（合成データの確認用）。セッションは 1 時間ずつずらす。"""
    beat_rows: list[pd.DataFrame] = []
    session_rows = []
    for i, r in enumerate(recordings):
        start = pd.Timestamp(origin + timedelta(hours=i))
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
            }
        )
    pd.concat(beat_rows).to_csv(beats_path, index=False)
    pd.DataFrame(session_rows).to_csv(sessions_path, index=False)
