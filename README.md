# AEX2

**Array Exchange v2** — 広域ネットワーク上に分散した計算資源の間で、配列データを
部分的・オンデマンドに転送するためのデータ流通基盤。

Rust 実装で、メタデータ操作を担う**コントロールプレーン** (gRPC) と、実データを運ぶ
**データプレーン** (生 TCP の独自フレームプロトコル) を分離する。前身は
[`aex`](https://github.com/keichi/aex) (Python + gRPC)。

設計の全体像・プロトコル仕様・マイルストーンは [SPEC.md](SPEC.md) を参照のこと。

## 状態

**M2 (データプレーン最小構成) まで実装済み。** Rust クライアントから `.npy` の
選択を取得できる。実データは protobuf を一切通らず、カーネルから呼び出し側の
バッファへ直接読み込まれる。

- `aex-core` — `DType`、`ErrorClass` / `AexError`、選択の解決 (`Index` の正規化と
  `SelectionLayout`)、`ArrayFile` / `ArrayDataset` トレイト、`.npy` バックエンド
- `aex-wire` — データプレーンのワイヤ形式。ハンドシェイク、32 バイト固定ヘッダの
  フレーム、複数接続が 1 個の出力バッファへ書き込むための `ScatterBuffer`。
  サーバとクライアントが**同一コードから**エンコード/デコードする
- `aex-proto` — `protos/aex.proto` から tonic/prost が生成するコードと、
  生成型とコア型の相互変換
- `aex-server` — コントロールプレーン (セッション、ファイル、メタデータ、
  `PrepareSelection`)、`TransferRegistry`、接続ごとに読みスレッドと送りスレッドを
  持つデータプレーン
- `aex-client` — 同期 API の Rust クライアント。`read_selection_into` /
  `read_selection` / `read_selection_as`

Python バインディング (`aex-py` と `python/`) はまだない。`PrepareSelections`
(gather) と `ApplyFunction` は `UNIMPLEMENTED` を返す。

### この時点での制限

M4 以降で解消する予定の、**実装上の**制限 (プロトコルの制限ではない)。

- **連続な選択のみ**。`arr[:]` / `arr[5]` / `arr[10:20]` / `arr[[8,9,10]]` のように
  ソース上で 1 個の連続領域になる選択を扱う。`arr[:, 0:50]` や `arr[::2]` のような
  ストライド選択・断片的選択は `ErrorClass::Request` で明示的に拒否する
- **データ接続は 1 本**。チャンク分割はクライアントが行い、1 本の接続で逐次
  フェッチする。並列ストリーム・ワークスティーリング・credit パイプラインは未実装。
  なお localhost では単一接続で既にマシンの限界に達しており、接続を増やしても
  合計は増えない
- `tcp.sndbuf` / `tcp.congestion` は設定を受け付けるがまだ適用しない
  (起動時に警告を出す)

ローカル (同一ホスト) での転送性能の測定結果は `docs/` にある
([M2 時点](docs/benchmark-m2-local.md)、[ダブルバッファリング](docs/benchmark-double-buffering.md)、
[ストレージを外した場合](docs/benchmark-null-backend.md)、
[sendfile を採らない理由](docs/sendfile.md))。メモリ上のデータで単一接続
12,048 MiB/s (iPerf3 の 62 %)、ストレージを経路から外すと 15,791 MiB/s (同 81 %)、
ディスク上のデータでは `pread` の限界近く。

## 転送の流れ

```
クライアント                                            サーバ
  │ ── gRPC: PrepareSelection ──────────────────────→ │ 選択を解決
  │ ←── TransferPlan{request_id, ticket, total} ───── │ plan を登録
  │                                                    │
  │  出力バッファを確保し、論理バイト列をチャンクに分割 │
  │ ── FETCH{request_id, offset, len} + ticket ─────→ │ pread → バッファ
  │ ←── DATA{request_id, offset, len} + 生バイト ──── │ writev(header, buf)
```

選択が `inline_limit_bytes` (既定 64 KiB) 以下なら、サーバは `TransferPlan` に
データ本体を載せて返す。データプレーンを一切使わず 1 RTT で完結するため、小さい
対話的な取得が v1 より遅くなることがない。

すべてのオフセットは**論理バイト列** (選択結果を C 順で平坦化した仮想的なバイト列)
上の座標である。そのためチャンクはどの接続で受け取っても正しい位置に書け、失われた
チャンクだけを再取得できる。

## ビルドとテスト

`protoc` が必要である (`brew install protobuf` / `apt install protobuf-compiler`)。

```console
$ cargo test
$ cargo clippy --all-targets --all-features -- -D warnings
$ cargo fmt --all -- --check
```

デバッグとリリースで挙動が変わる箇所 (オーバーフロー検査) があるため、CI は
`cargo test` を両プロファイルで実行する。

## 使い方

```console
$ aex-server --control-addr 127.0.0.1:50051 --data-addr 127.0.0.1:50052 \
    --root /path/to/data
```

設定ファイルを渡すこともできる (指定しなかった項目は既定値のまま)。項目は
SPEC §8.2 を参照のこと。

```console
$ aex-server --config server.toml
```

```rust
use aex_client::{Client, ClientConfig, Index, Item};

let client = Client::connect("http://127.0.0.1:50051", ClientConfig::default())?;
let handle = client.open("ocean.npy")?;           // データルート配下の相対パス

if let Item::Dataset(info) = client.get_item(handle, "array")? {
    println!("{:?} {:?}", info.dtype, info.shape);
}

// 配列全体、または先頭軸の連続した範囲
let all = client.read_selection_as::<f32>(handle, "array", &[])?;
let rows = client.read_selection_as::<f32>(handle, "array", &[Index::range(10, 20)])?;
println!("{:?} {:.1} MiB/s", rows.shape, rows.transfer.throughput_mib_per_sec());

// 呼び出し側のバッファへ直接読む (コピーが 1 回も入らない経路)
let mut buf = vec![0u8; all.bytes.len()];
client.read_selection_into(handle, "array", &[], &mut buf)?;

client.disconnect()?;
```

ベンチマークのパラメータ掃引のため、主要な設定は環境変数でも上書きできる
(`AEX_STREAMS`、`AEX_CHUNK_BYTES`、`AEX_MAX_RETRIES`、`AEX_TCP_NODELAY` ほか)。

## 前提と制約

- 対象 OS は Linux (最適化対象) と macOS。`pread` を使うため Unix 系に限る
- `.npy` のみ対応。HDF5 / netCDF4 / Zarr は将来課題 (SPEC §14.1)
- 読み出し専用
- fortran order とビッグエンディアンの `.npy` は非対応 (SPEC §7.2)
- データプレーンは平文。認証はセッショントークンと転送ごとの ticket のみで、
  「少数クライアント・信頼できる環境」を前提とする。厳密なマルチテナント制御や
  暗号化は行わない (TLS は HELLO の `flags` に枠のみ確保)
- 接続ごとに専用 OS スレッドを使う。スレッド数は
  `クライアント数 × 接続数` に比例する

## リポジトリ構成

```
protos/aex.proto   コントロールプレーンの定義
crates/aex-core/   共通型・選択の解決・バックエンド (tokio / tonic に依存しない)
                   `.npy` と、転送経路の測定用の合成バックエンド
crates/aex-wire/   データプレーンのワイヤ形式 (サーバとクライアントで共用)
crates/aex-proto/  aex.proto から生成されるコードと型変換
crates/aex-server/ サーバ (両プレーン)
crates/aex-client/ Rust クライアント
benchmarks/        転送性能の測定 (基準値の iperf3 / pread と、端から端まで)
docs/              測定結果
tests/rust/        サーバとクライアントを同一プロセスで動かす統合テスト
```

M3 で `aex-py` と `python/` が加わる (SPEC §4)。
