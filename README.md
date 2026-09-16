# AEX2

**Array Exchange v2** — 広域ネットワーク上に分散した計算資源の間で、配列データを
部分的・オンデマンドに転送するためのデータ流通基盤。

Rust 実装で、メタデータ操作を担う**コントロールプレーン** (gRPC) と、実データを運ぶ
**データプレーン** (生 TCP の独自フレームプロトコル) を分離する。前身は
[`aex`](https://github.com/keichi/aex) (Python + gRPC)。

設計の全体像・プロトコル仕様・マイルストーンは [SPEC.md](SPEC.md) を参照のこと。

## 状態

**M0 (足場) まで実装済み。** 現時点で動くのは `aex-core` の以下の部分である。

- `DType` — AEX の 14 要素型と numpy `descr` の相互変換
- `ErrorClass` / `AexError` — SPEC §5.6 の 6 クラスによるエラー分類
- `NpyFile` — `.npy` のヘッダ解析 (shape / dtype / データ開始位置) と、
  `pread` による論理バイト列の範囲読み出し

サーバ、クライアント、Python バインディングはまだない (M1 以降)。

## ビルドとテスト

```console
$ cargo test
$ cargo clippy --all-targets -- -D warnings
$ cargo fmt --all -- --check
```

デバッグとリリースで挙動が変わる箇所 (オーバーフロー検査) があるため、CI は
`cargo test` を両プロファイルで実行する。

## 前提と制約

- 対象 OS は Linux (最適化対象) と macOS。`pread` を使うため Unix 系に限る
- `.npy` のみ対応。HDF5 / netCDF4 / Zarr は将来課題 (SPEC §14.1)
- 読み出し専用
- fortran order とビッグエンディアンの `.npy` は非対応 (SPEC §7.2)
- 「少数クライアント・信頼できる環境」を前提とする。厳密なマルチテナント制御は行わない

## リポジトリ構成

```
crates/aex-core/   共通型とバックエンド (tokio / tonic に依存しない)
```

M1 以降で `aex-wire` / `aex-proto` / `aex-server` / `aex-client` / `aex-py` と
`python/` / `protos/` / `benchmarks/` が加わる (SPEC §4)。
