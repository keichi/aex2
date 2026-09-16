# AEX2

**Array Exchange v2** — 広域ネットワーク上に分散した計算資源の間で、配列データを
部分的・オンデマンドに転送するためのデータ流通基盤。

Rust 実装で、メタデータ操作を担う**コントロールプレーン** (gRPC) と、実データを運ぶ
**データプレーン** (生 TCP の独自フレームプロトコル) を分離する。前身は
[`aex`](https://github.com/keichi/aex) (Python + gRPC)。

設計の全体像・プロトコル仕様・マイルストーンは [SPEC.md](SPEC.md) を参照のこと。

## 状態

**M1 (コントロールプレーン) まで実装済み。** Rust クライアントからサーバに接続し、
`.npy` のメタデータを取得できる。

- `aex-core` — `DType` (14 要素型と numpy `descr` の相互変換)、`ErrorClass` /
  `AexError` (SPEC §5.6 の 6 クラス)、`ArrayFile` / `ArrayDataset` トレイト、
  `.npy` バックエンド (ヘッダ解析と `pread` による範囲読み出し)
- `aex-proto` — `protos/aex.proto` から tonic/prost が生成するコード。proto は
  SPEC §5 の全定義を含む
- `aex-server` — `Connect` / `Disconnect` / `OpenFile` / `CloseFile` /
  `GetItem` / `ListChildren`、セッションとファイルのレジストリ、データルート
  によるパス制限、SPEC §8.2 の設定ファイル
- `aex-client` — 同期 API の Rust クライアント (内部に tokio ランタイムを持つ)

データ転送 (`PrepareSelection` / データプレーン) と Python バインディングは
まだない。転送系の RPC は `UNIMPLEMENTED` を返す。

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
$ aex-server --control-addr 127.0.0.1:50051 --root /path/to/data
```

設定ファイルを渡すこともできる (指定しなかった項目は既定値のまま)。項目は
SPEC §8.2 を参照のこと。

```console
$ aex-server --config server.toml
```

```rust
use aex_client::{Client, ClientConfig, Item};

let client = Client::connect("http://127.0.0.1:50051", ClientConfig::default())?;
let handle = client.open("ocean.npy")?;           // データルート配下の相対パス
if let Item::Dataset(info) = client.get_item(handle, "array")? {
    println!("{:?} {:?}", info.dtype, info.shape);
}
client.disconnect()?;
```

## 前提と制約

- 対象 OS は Linux (最適化対象) と macOS。`pread` を使うため Unix 系に限る
- `.npy` のみ対応。HDF5 / netCDF4 / Zarr は将来課題 (SPEC §14.1)
- 読み出し専用
- fortran order とビッグエンディアンの `.npy` は非対応 (SPEC §7.2)
- 「少数クライアント・信頼できる環境」を前提とする。厳密なマルチテナント制御は行わない

## リポジトリ構成

```
protos/aex.proto   コントロールプレーンの定義
crates/aex-core/   共通型とバックエンド (tokio / tonic に依存しない)
crates/aex-proto/  aex.proto から生成されるコード
crates/aex-server/ サーバ (コントロールプレーン)
crates/aex-client/ Rust クライアント
tests/rust/        サーバとクライアントを同一プロセスで動かす統合テスト
```

M2 以降で `aex-wire` (データプレーンのフレーム)、M3 で `aex-py` と `python/` が
加わる (SPEC §4)。
