# M3: v1 と v2 の比較 (Python から、同一ホスト)

M3 で Python バインディングが揃い、v1 と v2 を**同じスクリプト・同じファイル**で
比べられるようになった (SPEC §13 M3)。以降の最適化の効果はこの値を起点に追う。

## 条件

- Apple M4、メモリ 32 GiB、macOS。サーバとクライアントは同一ホスト (loopback)
- v1: `../aex` の `4dcb6c3`、Python 3.13 + grpcio、`python -m aex.server`
- v2: M3 時点、リリースビルド (`aex-server` と `maturin develop --release`)。
  データ接続 1 本、チャンクはサーバ推奨値
- データ: `benchmarks/create_benchmark_data.py` が書く 2 次元 float32 の `.npy`。
  書いた直後なのでページキャッシュに載っている
- 計測: `benchmarks/run_benchmarks.py --runs 5` で `arr[:]` 1 回の所要時間。表は 5 回の平均

```console
$ benchmarks/run_benchmarks.py --sizes 10M 100M 1G --runs 5            # v2
$ uv run --project ../aex python benchmarks/run_benchmarks.py \
    --v1 ../aex --sizes 10M 100M 1G --runs 5                           # v1
```

## 結果

| サイズ | v1 (MiB/s) | v2 (MiB/s) | v2 / v1 |
|---|---:|---:|---:|
| 10 MiB | 2,354 | 11,353 | 4.8 |
| 100 MiB | 2,467 | 9,825 | 4.0 |
| 1 GiB | 2,416 | 10,740 | 4.4 |

v2 は単一接続のまま v1 の約 4.4 倍である。Python 層を通しても、Rust クライアント単体の
M2 時点の値 ([benchmark-m2-local.md](benchmark-m2-local.md)) と同じ水準に収まっており、
`np.empty` した配列へ直接受信する経路にコピーが増えていないことと整合する。

v1 はサイズによらず約 2.4 GiB/s で頭打ちになる。gRPC ストリームの 1 MiB 断片ごとに
protobuf のデコードと `bytearray` への再コピーが入るためである (SPEC §1.3)。
