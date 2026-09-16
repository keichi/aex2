# M2 転送性能測定 (localhost)

M2 (データプレーン最小構成、単一接続) の転送性能を、同一ホスト上で測定した記録である。
測定日 2026-09-16、対象コミットは `43d5040` (M2 完了時点)。

再現手順は [ベンチマークの回し方](#ベンチマークの回し方) に、生の出力は
[付録](#付録-生の出力) にある。

> **続き**: ここで「読みと送りが逐次であること」を律速として挙げた点は、その後
> ダブルバッファリングを実装して解消した。メモリ上のデータは 7,966 →
> 12,048 MiB/s になっている。[ダブルバッファリングの効果](benchmark-double-buffering.md)
> を参照のこと。

## 要約

| 条件 | AEX2 | 上限 (pread) | 到達率 |
|---|---:|---:|---:|
| **ディスク上のデータ** (ページキャッシュ退避済み) | **3,077 MiB/s** | 3,089 MiB/s | **99.6 %** |
| **メモリ上のデータ** (RAM ディスク) | **7,151 MiB/s** | 16,481 MiB/s | **43 %** |
| メモリ上・4 クライアント同時 (接続 4 本相当) | 12,912 MiB/s | 18,249 MiB/s | 71 % |

ループバック TCP の上限 (iperf3) は 17,539〜19,480 MiB/s (147〜163 Gbit/s)。

結論は 2 つある。

1. **ディスク上のデータでは、AEX2 はストレージの限界そのものに達している。** `pread`
   だけを回した場合と 0.4 % しか違わない。この経路に改善の余地はない。
2. **メモリ上のデータでは単一接続で上限の 43 % に留まる。** これは M2 が意図的に
   省いている 2 つの重なり — サーバの読みと送りの分離 (ダブルバッファリング) と、
   クライアントの credit パイプライン — が欠けているためで、下の
   [どこで律速しているか](#どこで律速しているか) で定量的に確認できる。同じホストで
   4 クライアントを同時に走らせると合計 12,912 MiB/s まで伸びるので、**律速は
   アーキテクチャではなく単一ストリームの直列化**である。

実用上の含意として、**10 GbE (1,192 MiB/s) なら現状の単一接続でもリンクの 6 倍を
出しており、律速はネットワーク側にある**。単一ストリームの直列化が問題になるのは
100 GbE 級か、今回のようなループバックに限られる。

## 測定環境

| 項目 | 値 |
|---|---|
| ホスト | Mac16,12 (Apple M4, 4 P-core + 6 E-core), 32 GiB RAM |
| OS | macOS 26.6.2 (25G83), Darwin 25.6.0 |
| ストレージ | APFS on Apple Fabric NVMe (内蔵 SSD) |
| Rust | 1.98.1、`--release` (`lto = "thin"`, `codegen-units = 1`) |
| iperf3 | 3.21 |
| AEX2 | `43d5040`、既定設定 (`default_chunk_bytes` 4 MiB、`max_fetch_bytes` 16 MiB、`inline_limit_bytes` 64 KiB) |

サーバとクライアントは同一ホストの別プロセスで、127.0.0.1 で通信する。

## 測定方法

### データの置き場所

**メモリ上**は 8 GiB の RAM ディスク (HFS+) に 4 GiB の `float32` 配列を置いた。
読み出しは必ずメモリコピーになる。

**ディスク上**は SSD 上に **48 GiB** (RAM の 1.5 倍) の配列を作り、その**先頭 4 GiB**
を転送対象とした。各測定の前に**後半 40 GiB を読んで先頭を追い出す**ことでコールド
キャッシュを作る。macOS の `purge` は root を要求するため、この方法を採った。
退避が効いていることは `pread` 自体の数字で確認できる (ウォーム 16,800 MiB/s に対し
コールド 2,900〜3,140 MiB/s と 5 倍以上の差)。

配列は 0 から数え上げた `float32` で埋めてある。転送後に先頭と末尾の値を検査して
いるので、チャンクが 1 個ずれた転送は「それらしい数字」ではなく不一致として落ちる。

### 何と比べるか

3 つの上限を測り、AEX2 をそれに対して位置づける。

- **iperf3 (ループバック)** — ソケットをバイトが通過できる速度。ただし**ループバック
  なので実質メモリ帯域であり、ネットワークの測定ではない**。実ネットワークでの測定は
  別途必要である (SPEC §12.2)。
- **`pread`** — ストレージからバイトが出てくる速度。サーバの読み経路と同じ形
  (1 個のバッファへ繰り返し `pread`) で、余計なものが何も入っていない。
- **AEX2** — 実際にクライアントが得る速度。

スループットは中央値で報告する。マシン上の別の何かで 1 回遅れた測定が、記録に残る
数字を動かすべきではないため。あわせて **GiB あたりの CPU 秒**を報告する。ループ
バックではスループットは設計よりメモリバスの性質を語るので、その速度を出すのに
マシンをどれだけ使ったかのほうが正直な指標になる。

## 結果

### 基準値

```
iperf3 -l 128K                                       17539 MiB/s (147.1 Gbit/s)
iperf3 -l 256K                                       18448 MiB/s (154.8 Gbit/s)
iperf3 -l 1M                                         19480 MiB/s (163.4 Gbit/s)

pread memory    chunk=1 MiB                          16810 MiB/s   cpu 0.06 s/GiB
pread memory    chunk=4 MiB                          18249 MiB/s   cpu 0.06 s/GiB
pread disk-cold chunk=2 MiB                           3140 MiB/s   cpu 0.11 s/GiB
pread disk-cold chunk=4 MiB                           3089 MiB/s   cpu 0.11 s/GiB
```

ループバック TCP と メモリ上の `pread` がほぼ同じ 17〜19 GiB/s に並ぶ。どちらも
同じメモリバスを使っているためで、**localhost ではネットワークとストレージが同じ
上限を共有する**。コールドの SSD だけがその 1/6 のところにいる。

### AEX2、チャンクサイズ掃引

| チャンク | メモリ上 | ディスク上 (コールド) |
|---:|---:|---:|
| 256 KiB | 4,606 MiB/s | 2,271 MiB/s |
| 1 MiB | 6,737 MiB/s | 2,218 MiB/s |
| **2 MiB** | **7,151 MiB/s** | 2,701 MiB/s |
| **4 MiB** | 4,440 MiB/s (best 6,139) | **3,077 MiB/s** |
| 16 MiB | 4,031 MiB/s | 2,412 MiB/s |

- メモリ上の最適は **2 MiB**、ディスク上の最適は **4 MiB**。既定の 4 MiB は
  ディスク上のデータには合っているが、メモリ上のデータには大きすぎる。
- 16 MiB は両方で悪化する。1 回のフェッチが「読み切ってから送り切る」形になるため、
  チャンクが大きいほど重なりの無さがそのまま効く。
- メモリ上の 4 MiB は中央値 4,440 に対し最良 6,139 と振れが大きい。P コアと E コアの
  どちらに乗るかの影響と見られる。

### 同時クライアント数によるスケーリング (メモリ上)

各クライアントが 1 本ずつ接続を張るので、**1 クライアントが複数接続を張れるように
なった (M4) ときに到達しうる値**の目安になる。

| 同時クライアント数 | 1 本あたり | 合計 | 上限 (18,249) に対して |
|---:|---:|---:|---:|
| 1 | 7,000 MiB/s | 7,000 MiB/s | 38 % |
| 2 | 5,888 MiB/s | 11,775 MiB/s | 65 % |
| **4** | 3,228 MiB/s | **12,912 MiB/s** | **71 %** |
| 6 | 1,919 MiB/s | 11,512 MiB/s | 63 % |
| 8 | 1,414 MiB/s | 11,314 MiB/s | 62 % |

**接続を増やせば合計は 1.8 倍まで伸び、4 本で頭打ちになる。** 単一接続の 7,000 MiB/s
がアーキテクチャの限界ではないことの直接の証拠である。6 本以降で合計が落ちるのは、
GiB あたり CPU が 0.07 → 0.21 s/GiB に増えていることから、10 コアに対して
`接続数 × 2` のスレッドが多すぎるためと見られる。

### レイテンシ (小さい取得)

```
GetItem (メタデータのみ)      p50    36.5 us  p99   108.9 us
read      4 B                p50    33.7 us  p99    51.6 us   inline, 1 往復
read   1 KiB                 p50    42.9 us  p99    61.7 us   inline, 1 往復
read  16 KiB                 p50    46.3 us  p99    65.8 us   inline, 1 往復
read  64 KiB                 p50    57.8 us  p99    77.5 us   inline, 1 往復
read  64 KiB + 4 B           p50    58.0 us  p99    78.9 us   データプレーン, 2 往復
read 256 KiB                 p50    74.4 us  p99    91.8 us   データプレーン, 2 往復
read   1 MiB                 p50   127.2 us  p99   153.8 us   データプレーン, 2 往復
```

- **4 バイトの取得は 33.7 µs で、メタデータ RPC 1 回 (36.5 µs) と同じ。** inline 返却は
  意図どおり「小さい取得は RPC 1 往復ぶん」を達成している。
- **一方で、閾値の 64 KiB では inline (57.8 µs) とデータプレーン (58.0 µs) に差がない。**
  ループバックの往復が 20 µs 程度しかないため、節約した 1 往復ぶんを、64 KiB を
  protobuf に通すコピーが食い潰している。
- inline 経路の価値は「RTT が支配的な環境で 2 RTT が 1 RTT になること」であり、
  **RTT が 20 µs の環境では原理的に実証できない**。WAN またはエミュレートした遅延で
  改めて測る必要がある。
- 同時に、**既定の 64 KiB という閾値は低遅延リンクには大きすぎる**ことも分かる。
  16 KiB では inline が 46.3 µs で明確に有利なままである。閾値は SPEC §12 の方針どおり
  対象ネットワーク上の実測で決めるべきで、localhost だけを見て決めてはならない。

### CPU 効率

| | GiB あたり CPU |
|---|---:|
| クライアント (メモリ上、2 MiB チャンク) | 0.07 s/GiB |
| サーバ (同、掃引 100 GiB で 15.2 s) | 0.15 s/GiB |
| 合計 | **0.22 s/GiB** |

7,151 MiB/s = 6.98 GiB/s で走らせているときに 1.5 コア相当。10 コアのうち 15 % である。

## どこで律速しているか

チャンクサイズ掃引から 1 チャンクあたりの所要時間を出すと、素直な直線が引ける。

| チャンク | 実測スループット | 1 チャンクあたり |
|---:|---:|---:|
| 256 KiB | 4,606 MiB/s | 54.3 µs |
| 1 MiB | 6,737 MiB/s | 148.4 µs |
| 2 MiB | 7,151 MiB/s | 279.7 µs |

3 点の最小二乗で限界帯域と固定費に分けると、

```
限界帯域 ≈ 7,760 MiB/s        固定費 ≈ 21 µs / チャンク
```

この 2 つがそれぞれ M2 が省いた最適化に対応する。

**限界帯域 7,760 MiB/s は、メモリ帯域 16,500 MiB/s のほぼ半分である。** サーバが
「`pread` で読み切る → `writev` で送り切る」を逐次に行っているためである。読みと
送りはどちらもメモリ帯域で動くので、重ねなければ実効は半分になる。SPEC §6.5.3 の
ダブルバッファリング (M4) がここに効く。

**固定費 21 µs = パイプライン化されていない往復。** クライアントは `FETCH` を投げて
`DATA` を待ってから次の `FETCH` を投げる。credit ベースのパイプライン (SPEC §6.4、M4)
がここに効き、チャンクが小さいほど影響が大きい (256 KiB では所要時間の 39 % が固定費)。

この読みは同時クライアント数のスケーリングとも整合する。接続 4 本の合計
12,912 MiB/s は、単一接続 7,000 MiB/s の 1.8 倍であり、「直列化されている 2 つの
処理を重ねれば最大 2 倍」という上の見立てとほぼ一致する。

ディスク上のデータでこの問題が見えないのは、SSD の 3,100 MiB/s が上の 7,760 MiB/s
よりずっと低く、**読みが完全に律速しているから**である。ネットワークが十分速ければ
M2 の実装でもストレージの限界に張り付く。

## M4 に向けて

この測定から、M4 (並列化と I/O 最適化) の優先順位は次のようになる。

1. **ダブルバッファリング (サーバの読み/送り分離)** — 単一接続で最大 2 倍。上の
   分解が示すとおり、これが一番大きい。
2. **複数接続とワークスティーリング** — 4 本で合計 12,912 MiB/s まで確認済み。
   ただし 6 本以降は CPU 律速で逆効果なので、**既定値は 4 が妥当**という実測の裏付けが
   得られた。
3. **credit パイプライン** — localhost では固定費 21 µs ぶんだが、RTT が ms 単位の
   WAN では支配的になる。効果の確認には遅延のある環境が要る。
4. 既定チャンクサイズは、メモリ上なら 2 MiB、ディスク上なら 4 MiB。**転送ごとに
   データがどこにあるかは分からない**ので、既定 4 MiB のままにして、M4 後に
   ダブルバッファリングが入った状態で掃引し直すのがよい。

## この測定で分かっていないこと

- **実ネットワークでの性能。** ループバックはメモリバスであって、ネットワークでは
  ない。10 GbE / 100 GbE および WAN での測定は別途必要 (SPEC §12.2)。
- **v1 (Python 実装) との比較。** Python バインディング (M3) が要る。
- **inline 経路の効果。** RTT が 20 µs の環境では原理的に測れない。
- **ストライド選択・fancy 選択の性能。** M2 は連続選択しか解決しないため、断片ごとの
  `pread` が syscall 律速になるかどうか (SPEC §6.5.2 の未決事項) はまだ測れていない。
- **コールドキャッシュ測定の n = 1。** 1 回の測定に 40 GiB の退避読みが要るため
  繰り返していない。ウォーム側の再現性から見て ±5 % 程度と考えられる。

## ベンチマークの回し方

```console
$ benchmarks/run-local.sh setup      # ビルド、RAM ディスク作成、データ生成 (48 GiB)
$ benchmarks/run-local.sh run        # 測定
$ benchmarks/run-local.sh teardown   # RAM ディスクとデータを削除
```

`setup` は RAM の 1.5 倍のファイルを作るので、空き容量に注意すること (32 GiB RAM の
マシンで 48 GiB)。macOS 専用である。Linux では RAM ディスクを tmpfs に、キャッシュ
退避を `/proc/sys/vm/drop_caches` に置き換えればよい。

個別の道具は単独でも使える。

```console
$ cargo build --release -p aex-bench
$ target/release/aexbench http://127.0.0.1:50151 arr.npy \
      --bytes $((1<<30)) --chunk $((2<<20)) --reps 5
$ target/release/preadbench /path/to/arr.npy --bytes $((1<<30)) --chunk $((2<<20))
$ target/release/latbench http://127.0.0.1:50151 arr.npy
```

## 付録: 生の出力

<details>
<summary><code>benchmarks/run-local.sh run</code> の出力 (2026-09-16)</summary>

```
### environment
Mac16,12, Apple M4, 10 cores, 32 GiB RAM
ProductName:		macOS ProductVersion:		26.6.2 BuildVersion:		25G83 
aex-server 0.1.0

### baseline: loopback TCP (iperf3)
iperf3 -l 128K                                       17539 MiB/s (147.1 Gbit/s)
iperf3 -l 256K                                       18448 MiB/s (154.8 Gbit/s)
iperf3 -l 1M                                         19480 MiB/s (163.4 Gbit/s)

### baseline: pread, data resident in memory (RAM disk)
pread memory chunk=262144                    median    16430 MiB/s (137.8 Gbit/s)  best    16574  cpu  0.07 s/GiB  n=3
pread memory chunk=1048576                   median    16810 MiB/s (141.0 Gbit/s)  best    17043  cpu  0.06 s/GiB  n=3
pread memory chunk=2097152                   median    16481 MiB/s (138.2 Gbit/s)  best    16612  cpu  0.06 s/GiB  n=3
pread memory chunk=4194304                   median    18249 MiB/s (153.1 Gbit/s)  best    18347  cpu  0.06 s/GiB  n=3
pread memory chunk=16777216                  median    12541 MiB/s (105.2 Gbit/s)  best    13981  cpu  0.08 s/GiB  n=3

### baseline: pread, data on disk (page cache evicted before each run)
pread disk-cold chunk=262144                 median     2989 MiB/s ( 25.1 Gbit/s)  best     2989  cpu  0.11 s/GiB  n=1
pread disk-cold chunk=1048576                median     2965 MiB/s ( 24.9 Gbit/s)  best     2965  cpu  0.10 s/GiB  n=1
pread disk-cold chunk=2097152                median     3140 MiB/s ( 26.3 Gbit/s)  best     3140  cpu  0.11 s/GiB  n=1
pread disk-cold chunk=4194304                median     3089 MiB/s ( 25.9 Gbit/s)  best     3089  cpu  0.11 s/GiB  n=1
pread disk-cold chunk=16777216               median     3106 MiB/s ( 26.1 Gbit/s)  best     3106  cpu  0.13 s/GiB  n=1

### AEX2, data resident in memory (RAM disk)
aex memory chunk=262144                      median     4606 MiB/s ( 38.6 Gbit/s)  best     4620  cpu  0.12 s/GiB  n=5
aex memory chunk=1048576                     median     6737 MiB/s ( 56.5 Gbit/s)  best     6987  cpu  0.07 s/GiB  n=5
aex memory chunk=2097152                     median     7151 MiB/s ( 60.0 Gbit/s)  best     7276  cpu  0.07 s/GiB  n=5
aex memory chunk=4194304                     median     4440 MiB/s ( 37.2 Gbit/s)  best     6139  cpu  0.11 s/GiB  n=5
aex memory chunk=16777216                    median     4031 MiB/s ( 33.8 Gbit/s)  best     4818  cpu  0.12 s/GiB  n=5
server cpu over the sweep: 15.2 s

### AEX2, data on disk (page cache evicted before each run)
aex disk-cold chunk=262144                   median     2271 MiB/s ( 19.1 Gbit/s)  best     2271  cpu  0.14 s/GiB  n=1
aex disk-cold chunk=1048576                  median     2218 MiB/s ( 18.6 Gbit/s)  best     2218  cpu  0.15 s/GiB  n=1
aex disk-cold chunk=2097152                  median     2701 MiB/s ( 22.7 Gbit/s)  best     2701  cpu  0.12 s/GiB  n=1
aex disk-cold chunk=4194304                  median     3077 MiB/s ( 25.8 Gbit/s)  best     3077  cpu  0.12 s/GiB  n=1
aex disk-cold chunk=16777216                 median     2412 MiB/s ( 20.2 Gbit/s)  best     2412  cpu  0.12 s/GiB  n=1

### AEX2, one connection each, several clients at once (memory)
aex memory[1/1] chunk=1048576                median     7000 MiB/s ( 58.7 Gbit/s)  best     7263  cpu  0.07 s/GiB  n=20

aex memory[1/2] chunk=1048576                median     5890 MiB/s ( 49.4 Gbit/s)  best     6175  cpu  0.08 s/GiB  n=20
aex memory[2/2] chunk=1048576                median     5885 MiB/s ( 49.4 Gbit/s)  best     6175  cpu  0.08 s/GiB  n=20

aex memory[4/4] chunk=1048576                median     3233 MiB/s ( 27.1 Gbit/s)  best     3376  cpu  0.15 s/GiB  n=20
aex memory[2/4] chunk=1048576                median     3240 MiB/s ( 27.2 Gbit/s)  best     3379  cpu  0.15 s/GiB  n=20
aex memory[3/4] chunk=1048576                median     3229 MiB/s ( 27.1 Gbit/s)  best     3388  cpu  0.15 s/GiB  n=20
aex memory[1/4] chunk=1048576                median     3210 MiB/s ( 26.9 Gbit/s)  best     3383  cpu  0.15 s/GiB  n=20

aex memory[4/6] chunk=1048576                median     1935 MiB/s ( 16.2 Gbit/s)  best     2133  cpu  0.20 s/GiB  n=20
aex memory[3/6] chunk=1048576                median     1914 MiB/s ( 16.1 Gbit/s)  best     2159  cpu  0.21 s/GiB  n=20
aex memory[2/6] chunk=1048576                median     1914 MiB/s ( 16.1 Gbit/s)  best     2134  cpu  0.21 s/GiB  n=20
aex memory[6/6] chunk=1048576                median     1924 MiB/s ( 16.1 Gbit/s)  best     2123  cpu  0.21 s/GiB  n=20
aex memory[1/6] chunk=1048576                median     1925 MiB/s ( 16.1 Gbit/s)  best     2101  cpu  0.21 s/GiB  n=20
aex memory[5/6] chunk=1048576                median     1901 MiB/s ( 15.9 Gbit/s)  best     2150  cpu  0.21 s/GiB  n=20

aex memory[1/8] chunk=1048576                median     1433 MiB/s ( 12.0 Gbit/s)  best     1665  cpu  0.21 s/GiB  n=20
aex memory[5/8] chunk=1048576                median     1412 MiB/s ( 11.8 Gbit/s)  best     1627  cpu  0.20 s/GiB  n=20
aex memory[7/8] chunk=1048576                median     1419 MiB/s ( 11.9 Gbit/s)  best     1647  cpu  0.21 s/GiB  n=20
aex memory[3/8] chunk=1048576                median     1407 MiB/s ( 11.8 Gbit/s)  best     1592  cpu  0.21 s/GiB  n=20
aex memory[2/8] chunk=1048576                median     1404 MiB/s ( 11.8 Gbit/s)  best     1563  cpu  0.21 s/GiB  n=20
aex memory[8/8] chunk=1048576                median     1396 MiB/s ( 11.7 Gbit/s)  best     1542  cpu  0.21 s/GiB  n=20
aex memory[4/8] chunk=1048576                median     1400 MiB/s ( 11.7 Gbit/s)  best     1553  cpu  0.21 s/GiB  n=20
aex memory[6/8] chunk=1048576                median     1443 MiB/s ( 12.1 Gbit/s)  best     1527  cpu  0.21 s/GiB  n=20

### AEX2, latency of a small read (memory)
GetItem (metadata only)    p50    36.5 us  p99   108.9 us
read 4 B                   p50    33.7 us  p99    51.6 us   inline, 1 round trip
read 1024 B                p50    42.9 us  p99    61.7 us   inline, 1 round trip
read 16384 B               p50    46.3 us  p99    65.8 us   inline, 1 round trip
read 65536 B               p50    57.8 us  p99    77.5 us   inline, 1 round trip
read 65540 B               p50    58.0 us  p99    78.9 us   data plane, 2 round trips
read 262144 B              p50    74.4 us  p99    91.8 us   data plane, 2 round trips
read 1048576 B             p50   127.2 us  p99   153.8 us   data plane, 2 round trips
```

</details>
