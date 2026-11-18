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

### 追跡・出来事・環境（省略可。docs/pilot-protocol.md の 4.2・7 章）
セッションの CSV に、追跡の時刻の列を足せます。3 つそろえて書くか、どれも書かないかにします（一部だけは止める）。

| 列 | 型 | 説明 |
| --- | --- | --- |
| `contact_end` | 日時 or 秒 | 接触の終わり |
| `track_start` | 日時 or 秒 | 戻った後、オフィスで追跡を始めた時刻 |
| `track_end` | 日時 or 秒 | 追跡の終わり |

順は `contact_start ≦ contact_end ≦ track_start ≦ track_end`。追跡の拍は、拍の CSV に続けて入れます。

出来事の CSV（`read_recordings(..., events_path=...)`）: `session, time, event`。`event` は `talk`・`phone`・`food`・`away` のどれか。その時刻を含む追跡の窓は解析から除きます。セッションの CSV に無い session は止めます。

環境の CSV（`read_environment`）: `session, time, ta_c, rh_pct, wind_ms, globe_c, laeq_db`（1 分ごと）。`contact_environment` で接触中の平均を出し、`pet_c`・`noise_db` をセッションの CSV に書く元にします。`contact_pet` は接触中の平均の気温・湿度・風・黒球温度から PET を出します。MRT は黒球温度から ISO 7726 の強制対流の式で求め（黒球の直径 0.15 m・放射率 0.95、`vmgs.thermal.globe_mrt_c`）、風は人体の高さの実測とみなして換算しません（下限 0.1 m/s）。PET の人体の条件はシミュレーターと同じ（1.37 met・0.5 clo）。1 分ごとの PET の平均ではなく、10 分の平均の環境から 1 回計算します。

```csv
session,subject,condition,contact_start,green,pet_c,noise_db,contact_end,track_start,track_end
P01-sit-leave-1,P01,sit-leave,2026-07-01T10:03:00+09:00,0.72,31.4,58,2026-07-01T10:13:00+09:00,2026-07-01T10:18:00+09:00,2026-07-01T11:13:00+09:00
```
```csv
session,time,event
P01-sit-leave-1,2026-07-01T10:41:20+09:00,phone
```

## 解析が前提にしていること
- 接触前に 180 秒以上の安静（基準）、接触後に 600 秒の記録があること（`study.session_curve` の既定）。
- 外れ値の RR（300〜2000 ms の外、前後 11 拍の中央値から 20% 以上離れたもの）は除き、除いた数を `SessionCurve.n_removed` に残す。
- 形式が合わないとき（必須の列がない、知らない条件名、session の重複、追跡の時刻の一部だけ・順が逆、知らない出来事、セッションの CSV に無い session の出来事）は `RecordingFormatError` で止める。
- 追跡の解析（`tracking.tracking_curve`）は、接触の終わりから 5〜60 分の 1 分窓を使う。帰りの 5 分、追跡の区間からはみ出す窓、出来事のある窓は NaN にする。
