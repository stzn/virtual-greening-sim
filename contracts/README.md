# データ契約 v1（舞台 ↔ 役者）

[tokyo-green-simulator](https://github.com/stzn/tokyo-green-simulator)（舞台: 都市緑化デジタルツイン）と、本リポジトリ（役者: 仮想移動緑化シミュレータ）の間で受け渡す JSON の形式です。全体の計画は [docs/integration-plan.md](../docs/integration-plan.md) にあります。

| ファイル | 向き | 中身 |
| --- | --- | --- |
| `stage.schema.json` | 舞台 → 役者 | 候補地点ごとの、時間別の日射遮蔽率、天空率、緑の文脈、需要（集計値） |
| `dispatch.schema.json` | 役者 → 舞台 | ポッドの時刻別の地点・行動・植物の健康度と、地点ごとの PET |
| `fixtures/*.sample.json` | — | 契約確認用に手で作ったサンプル（実測・実計算ではない） |

## 正本と更新の手順
- 正本は `src/vmgs/contracts.py`（Pydantic）です。スキーマを手で編集せず、モデルを直してから次を実行します。
  ```bash
  uv run python -m vmgs.contracts
  ```
- `contracts/` 一式（スキーマとサンプル）を、tokyo-green-simulator の `contracts/` へそのまま写します。両リポジトリのテストが、同じサンプルを同じスキーマで検証します。
- 互換性のない変更は `schema_version` を上げます（v1 を読む側は、知らない版を拒否します）。

## 決めごと
- 時刻は `Asia/Tokyo` の 0〜23 時。`hourly` の時刻と地点 ID は重複させない。
- 夜間（日射なし）の `shade_fraction` は 1.0 とする。
- 知らないキーは拒否する（綴りの誤りを黙って読み飛ばさない）。
- 需要（`demand`）には、地点・時刻ごとの集計値だけを入れる。個人の HRV などの計測値は入れない。
