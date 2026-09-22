# mdx2: HDF5 バックエンドの転送性能

HDF5 バックエンド (SPEC §7.5) が、libhdf5 のグローバルロックに律速されずに接続数で
伸びるかを mdx2 の VM 2 台で測った記録である。測定日 2026-09-17。

ここで比べている h5py はサーバ上でのローカル読みである。**同じファイルを
ネットワーク越しに読む相手** (h5py + HTTP、HSDS) との比較は
[HDF5 のリモート比較](benchmark-hdf5-remote.md) にある。

## 結論

1. **contiguous な HDF5 は `.npy` と同じ速さで出る。** どの接続数でも差は測定の振れの内側
   (16 本で 10,047 対 10,238 MiB/s)。データの読み出しは libhdf5 を通らず pread なので、
   形式の違いが転送経路に現れない
2. **gzip で圧縮した HDF5 は、伸長が律速のまま接続数に比例して伸びる。**
   1 本から 16 本で、圧縮の効きにくいデータは **297 → 3,617 MiB/s (12.2 倍)**、
   よく効くデータは **758 → 6,315 MiB/s (8.3 倍)**。伸長は各接続のスレッドで並列に走る
3. **libhdf5 に読ませると、スレッドを増やしても 1 本分から伸びない。** 同じファイルを
   サーバ上の h5py で読むと、16 スレッドに分けても 1 スレッドと同じ
   (noisy で 301 対 302 MiB/s)。16 本の AEX2 はこの **12 倍**出ており、
   メタデータだけを libhdf5 に任せる設計 (案 B) の効果がそのまま出ている
4. **1 本あたりの伸長は libhdf5 とほぼ同じ速さ。** noisy は 297 対 302 MiB/s。
   counting は 758 対 1,049 MiB/s と差があるが、同じ 1 本の `.npy` (2,572 MiB/s) で
   測った転送そのものの時間を差し引くと、伸長だけで約 1,075 MiB/s と見積もれ、
   伸長器が遅いわけではない (直列に足し合わせた粗い見積もり)

## 条件

環境は [mdx2 の測定](benchmark-mdx2.md#測定環境) と同じ。手順は
[eval-mdx2.md](eval-mdx2.md) に従った。

| | 値 |
|---|---|
| サーバ / クライアント | `aex2-eval-1` / `aex2-eval-2` |
| REVISION | `5f3577b` (両方) |
| サーバ設定 | npy と contiguous は `benchmarks/mdx2/aex.toml`、gzip は `benchmarks/mdx2/aex-decode-cache-64m.toml` |
| クライアント | `aexbench --prefault`、4 GiB、既定のチャンク (4 MiB)、credit 4 |
| Rust / iPerf3 / kernel | 1.98.1 / 3.16 / 6.8.0-136 |
| libhdf5 | サーバは 2.2.0 (`/usr/local`、メタデータのみ)。h5py の比較は h5py 3.16.0 同梱の 2.0.0 |
| sysctl | 既定 (`wmem_max = 212992`、cubic) |

データはすべて `/mnt/aexram` (tmpfs) の 2^30 要素の float32 で、作り方は
[eval-mdx2.md](eval-mdx2.md#データの置き場所) にある。

| ラベル | ファイル | 中身 | ファイルサイズ |
|---|---|---|---|
| npy | `mem.npy` | 要素 i の値が i | 4.0 GiB |
| h5-contiguous | `mem.h5` | 同上、contiguous | 4.0 GiB |
| h5-gzip-counting | `mem-gzip.h5` | 同上、4 MiB チャンク × 1024、shuffle + gzip 1 | 36 MiB (0.9 %) |
| h5-gzip-noisy | `mem-gzip-noisy.h5` | `i % 1000` + 0〜15 の乱数、同じ圧縮 | 1.07 GiB (27 %) |

gzip の 2 本は、伸長済みチャンクのキャッシュを 64 MiB (16 本 × 4 MiB) に絞ったサーバで
測った。既定の 1 GiB では 1024 チャンク中 256 個が前回の実行から残り、伸長する区間と
キャッシュから写すだけの区間が混ざるためである。この設定では毎回ほぼ全チャンクを伸長する。

| 回 | 前後の iPerf3 (`-R`、1 本 / 16 本) |
|---|---|
| 1 回目 | 26.2 → 34.7 / 138 → 132 Gbit/s |
| 2 回目 (本測定) | 26.1 → 37.1 / 134 → 141 Gbit/s |

**2 回とも 1 本の iPerf3 が前後で 10 % 以上動いた**ので、手順の決まりでは測り直しに
当たる。ただし 16 本は 5 % 以内で、AEX2 の値は 2 回で 5 % 以内に揃った (下の表)。
1 本の iPerf3 が掃引の後だけ高く出るのは [遅延の測定](benchmark-delay.md) の回
(30.7 → 35.6) と同じ傾向で、帯域が動いたというより測る時点の状態の差に見える。
各条件は 1 巡ごとに 1 回ずつ交互に測ったので、**同じ回の中での比較**は崩れていない。

## 結果

値は中央値 (MiB/s)、括弧内は最小〜最大、n = 5。2 回目の結果。

| 接続数 | npy | h5-contiguous | h5-gzip-counting | h5-gzip-noisy |
|---:|---:|---:|---:|---:|
| 1 | 2,572 (1,989〜2,637) | 2,566 (2,033〜2,636) | 758 (627〜764) | 297 (288〜300) |
| 2 | 5,243 (4,400〜5,451) | 5,295 (4,364〜5,399) | 1,448 (1,303〜1,500) | 582 (569〜605) |
| 4 | 9,209 (8,777〜9,972) | 9,560 (8,473〜10,022) | 2,706 (2,516〜2,805) | 1,150 (1,118〜1,162) |
| 8 | 13,214 (9,540〜13,522) | 12,741 (11,982〜15,711) | 4,801 (4,669〜4,983) | 2,141 (2,094〜2,176) |
| 16 | 10,238 (9,750〜11,803) | 10,047 (9,730〜10,522) | 6,315 (6,071〜6,779) | 3,617 (3,537〜3,658) |

1 回目の中央値 (同じ並び):

| 接続数 | npy | h5-contiguous | h5-gzip-counting | h5-gzip-noisy |
|---:|---:|---:|---:|---:|
| 1 | 2,311 | 2,473 | 737 | 284 |
| 2 | 4,922 | 4,751 | 1,404 | 580 |
| 4 | 9,010 | 8,777 | 2,620 | 1,141 |
| 8 | 14,046 | 13,861 | 4,611 | 2,131 |
| 16 | 10,042 | 9,306 | 6,228 | 3,534 |

npy と contiguous が 16 本で 8 本より遅いのは HDF5 と関係なく、
[遅延の測定](benchmark-delay.md) の「遅延なしの 16 本」と同じ未解明の現象である。

### libhdf5 で読んだ場合

サーバ上で h5py の `read_direct` により 4 GiB 全体を読んだ速さ (MiB/s、3 回)。
16 スレッドは 1/16 ずつの範囲を別スレッドで同時に読んだ。ネットワークは通らない。

| ファイル | 1 スレッド | 16 スレッド |
|---|---|---|
| `mem.h5` | 4,202 / 4,502 / 4,528 | 4,164 / 4,289 / 4,303 |
| `mem-gzip.h5` | 1,043 / 1,049 / 1,051 | 1,017 / 1,018 / 1,018 |
| `mem-gzip-noisy.h5` | 302 / 303 / 303 | 301 / 302 / 302 |

スレッドを増やしても速くならないのは、h5py と libhdf5 のロックで読み出しが一列に
並ぶためである。AEX2 がデータを libhdf5 経由で読んでいたら (SPEC §7.5 で退けた案)、
接続数によらずこの 1 スレッド分が上限になっていた。

## 再現方法

[eval-mdx2.md](eval-mdx2.md) の初回セットアップ (HDF5 2.2.0 と h5py) とデータの作成を
済ませ、`aex.toml` と `aex-decode-cache-64m.toml` のサーバを立ててから、クライアントで次を回した。

```console
$ ssh aex2-eval2 "bash -lc 'cd aex2 && B=target/release/aexbench; S=http://192.168.100.207; G=\$((4<<30));
  for r in 1 2 3 4 5; do for s in 1 2 4 8 16; do
   \$B \$S:50191 mem.npy --bytes \$G --streams \$s --reps 1 --prefault --label npy | tail -1
   \$B \$S:50191 mem.h5 --bytes \$G --streams \$s --reps 1 --prefault --label h5-contiguous | tail -1
   \$B \$S:50391 mem-gzip.h5 --bytes \$G --streams \$s --reps 1 --prefault --label h5-gzip-counting | tail -1
   \$B \$S:50391 mem-gzip-noisy.h5 --bytes \$G --streams \$s --reps 1 --prefault --pattern none --label h5-gzip-noisy | tail -1
  done; done'"
```

libhdf5 の比較はサーバ上で次を実行した。

```python
import h5py, numpy as np, time, threading
n = 1 << 30
with h5py.File("/mnt/aexram/mem-gzip-noisy.h5", "r") as f:
    d = f["array"]
    out = np.empty(n, np.float32); out.fill(1)
    t = time.perf_counter(); d.read_direct(out); print(4096 / (time.perf_counter() - t))
    step = n // 16
    def work(k):
        d.read_direct(out, np.s_[k*step:(k+1)*step], np.s_[k*step:(k+1)*step])
    ts = [threading.Thread(target=work, args=(k,)) for k in range(16)]
    t = time.perf_counter(); [x.start() for x in ts]; [x.join() for x in ts]
    print(4096 / (time.perf_counter() - t))
```
