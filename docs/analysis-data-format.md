# 実データ（RR 間隔）の形式（仮決め）

HRV の機器が未定のため、解析基盤（`src/vmgs/analysis/io.py` の `read_recordings`）が読む形式を仮に決めたもの。機器が決まったら、その機器の書き出しをこの形式に変換する部分を足す。

**個人の健康データはリポジトリに置かない・コミットしない。** 実データはリポジトリの外に置き、パスを渡して読む。

## ファイル
2 つの CSV（UTF-8、ヘッダー行あり）。

### 拍の CSV（1 行 1 拍）
| 列 | 型 | 説明 |
| --- | --- | --- |
| `subject` | 文字列 | 被験者 ID（仮名化したもの） |
| `session` | 文字列 | セッション ID（セッションの CSV と対応） |
| `timestamp` | 日時 or 秒 | その拍の時刻（RR 間隔の終わりの拍） |
| `rr_ms` | 数値 | RR 間隔 [ms] |

### セッションの CSV（1 行 1 セッション）
| 列 | 型 | 説明 |
| --- | --- | --- |
| `session` | 文字列 | セッション ID（重複不可） |
| `subject` | 文字列 | 被験者 ID |
| `condition` | 文字列 | `sit-stay`・`walk-stay`・`sit-leave`・`walk-leave` のどれか（歩行の有無 × オフィスからの離脱の有無） |
| `contact_start` | 日時 or 秒 | 条件を切り替えた（緑への接触を始めた）時刻。解析ではこれを 0 秒とする |
| `green` | 数値 | 緑の量（0〜1。緑視率など、割合で表したもの） |
| `pet_c` | 数値 | 体感温度 PET [°C]（シミュレーター側のフェーズ 2 の計算を使う予定） |
| `noise_db` | 数値 | 騒音 [dB(A)] |

### 時刻の表し方
`timestamp` と `contact_start` は、どちらも ISO 8601 の日時（例 `2026-07-01T10:03:12.345+09:00`。タイムゾーン付きを推奨、なしは UTC とみなす）か、どちらも同じ基準の秒の数値にする。

## 例
```csv
subject,session,timestamp,rr_ms
P01,P01-sit-leave-1,2026-07-01T10:00:00.812+09:00,812
P01,P01-sit-leave-1,2026-07-01T10:00:01.630+09:00,818
```
```csv
session,subject,condition,contact_start,green,pet_c,noise_db
P01-sit-leave-1,P01,sit-leave,2026-07-01T10:03:00+09:00,0.72,31.4,58
```

## 解析が前提にしていること
- 接触前に 180 秒以上の安静（基準）、接触後に 600 秒の記録があること（`study.session_curve` の既定）。
- 外れ値の RR（300〜2000 ms の外、前後 11 拍の中央値から 20% 以上離れたもの）は除き、除いた数を `SessionCurve.n_removed` に残す。
- 形式が合わないとき（必須の列がない、知らない条件名、session の重複）は `RecordingFormatError` で止める。
