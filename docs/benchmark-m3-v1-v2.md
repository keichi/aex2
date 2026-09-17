# M3: v1 と v2 の比較 (Python から)

M3 で Python バインディングが揃い、v1 と v2 を**同じスクリプト・同じファイル**で
比べられるようになった (SPEC §13 M3)。以降の最適化の効果はこの値を起点に追う。

## 同一ホスト (Mac)

### 条件

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

### 結果

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

## VM (mdx2)

[benchmark-mdx2.md](benchmark-mdx2.md) と同じ VM 2 台で、VM 内ループバックと VM 間を測った。
測定日 2026-09-17。

### 条件

- サーバは `aex2-eval-1`。「ローカル」はクライアントも `aex2-eval-1`、「VM 間」は
  クライアントが `aex2-eval-2` (RTT 平均 0.7 ms、MTU 1442)
- データは `aex2-eval-1` の tmpfs (`/mnt/aexram/m3`)。ファイルは Mac と同じスクリプトで
  生成し、サイズも同じ
- v1: `4dcb6c3`、Python 3.13.15、grpcio 1.67.1、numpy 2.1.3
- v2: M3 時点 (`835b747`)、リリースビルド、Python 3.13.15、numpy 2.5.3。
  データ接続 1 本、チャンクはサーバ推奨値 (4 MiB)。サーバ設定は既定値
- 計測は Mac と同じく `run_benchmarks.py --runs 5` の平均

### 結果

| サイズ | ローカル v1 | ローカル v2 | v2 / v1 | VM 間 v1 | VM 間 v2 | v2 / v1 |
|---|---:|---:|---:|---:|---:|---:|
| 10 MiB | 798 | 2,837 | 3.6 | 722 | 1,766 | 2.4 |
| 100 MiB | 518 | 2,548 | 4.9 | 431 | 1,884 | 4.4 |
| 1 GiB | 546 | 2,825 | 5.2 | 504 | 1,873 | 3.7 |

単位は MiB/s。**1 GiB で v2 は v1 の 5.2 倍 (ローカル)、3.7 倍 (VM 間)。**

- **v1 は VM では Mac の 4 分の 1 しか出ない** (約 2.4 GiB/s に対して 0.5 GiB/s)。
  ループバックでも VM 間でも差が小さく、ネットワークではなく v1 自身 (gRPC ストリームの
  デコードと再コピーを回す Python) が律速している。Mac との差は、Python の 1 スレッドが
  Sapphire Rapids の vCPU では M4 より遅いためと見ているが、プロファイルは取っていない
- **v2 の VM 間は 1.87 GiB/s**で、ワイヤの上限 (iPerf3 1 ストリーム 3,221 MiB/s) の 58 %。
  律速は前回の測定どおりサーバの送りスレッド 1 本と、チャンクごとの往復である
  ([benchmark-mdx2.md](benchmark-mdx2.md))。どちらも M4 (並列接続・credit) の対象
- 10 MiB は 1 回が 4〜14 ms と短く、初回のばらつき (接続の立ち上がり) が平均に入っている

### Python 層のコスト

同じ 1 GiB を、Rust クライアント単体 (`aexbench`、バッファを使い回す) と、Python から
出力配列を毎回 `np.empty` する場合 / 使い回す場合で比べた (`aexbench` は 1 次元の
`mem.npy` の先頭 1 GiB、Python は上の 2 次元ファイル)。5 回の中央値または平均。

| | ローカル | VM 間 |
|---|---:|---:|
| Rust (`aexbench`) | 3,380 | 2,247 |
| Python、バッファを使い回す | 3,291 | 2,157 |
| Python、毎回 `np.empty` (`arr[:]` と同じ) | 2,719 | 2,044 |

**Python 層そのものの費用は 3〜4 % に収まる。** 差の大半は、新しく確保した 1 GiB に
初めて書き込むときのページフォールトで、ローカルでは 17 % を占める (VM 間では転送の
ほうが遅いので 5 %)。M5 の `read_into` でバッファを使い回せば、この分は消える。

なお `run_benchmarks.py` による VM 間 1 GiB の値 (1,873) は、直後に測った上の
`np.empty` の値 (2,044) より 9 % 低い。同じ経路なので、時間帯による基盤側の揺らぎと見ている。
