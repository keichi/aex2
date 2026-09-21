# mdx2: HDF5 を遠くから読む 3 つのやり方の比較

[HDF5 バックエンドの測定](benchmark-hdf5.md) は AEX2 だけがネットワークを渡っていた。
[Zarr のリモート比較](benchmark-zarr-remote.md) と同じ問いが HDF5 にもある ──
**「HDF5 なら h5serv や HSDS で配れば済むのでは」**。これに答えるには、同じファイルを
同じリンクの向こうから、相手にも本気の設定で読ませる必要がある。

相手は 2 つ選んだ。

- **h5py + HTTP** ── nginx が `.h5` をそのまま配り、**クライアント側の libhdf5** が
  range request でシークしながら読む。特別なサーバが要らない、いちばんよくある形
- **HSDS + h5pyd** ── HDF Group 自身の REST サービス。**サーバ側で**読んで伸長し、
  値を返す。構造は AEX2 と同じ形である

h5serv は測っていない。**HDF Group 自身が 2022 年に sunset を宣言**しており
([Sunset for h5serv](https://www.hdfgroup.org/2022/10/31/sunset-for-h5serv/))、
要求を 1 本ずつしか処理しないシングルスレッドの実装なので、藁人形を殴ることになる。
後継が HSDS である。

測定日 2026-09-22。

## 結論

1. **HSDS には全区間で 2.25 〜 6.68 倍。** HSDS は AEX2 と**同じ構造** (サーバで読んで
   伸長し、値を送る) で、ワイヤに流すバイト数も同じ 4 GiB である。にもかかわらず
   近くで 2.5 〜 6.0 倍、100 ms でも 2.25 倍離れる。**設計の差ではなく実装の差**である
2. **HSDS は同じ仕事に 2 〜 5 倍の CPU を使う。** しかもサーバ側だけでなく
   クライアント側も 5 倍前後 (0.7 対 3.9 〜 4.4 s/GiB)。1 コアあたりで見ると、
   gzip を伸長しながら送る仕事は AEX2 が 566、HSDS が 200 MiB/s である
3. **h5py + HTTP のほうが HSDS より速い。** 近いところで 1.3 〜 3.8 倍。**HDF5 を
   リモートで読む相手として本当に強いのは、専用サービスではなく
   「ファイルをそのまま配って、クライアント側の libhdf5 に読ませる」形**である
4. **その h5py に対しては、無圧縮のファイルで 4.3 倍 (別の掃引では 4.8 倍)、
   圧縮の効かないファイルで 1.3 倍。** 前者は 4 GiB を range request で引くしか
   ないため、後者はワイヤのバイト数の差が 3.2 倍しかないためである
5. **よく圧縮できるファイルは、遠くなると h5py が勝つ。** `mem-gzip.h5` (0.9 %) では
   追加 RTT 25 ms で 0.96 倍、100 ms で 0.83 倍。**176 対 4,102 MiB** という
   ワイヤのバイト数が効く。[Zarr のとき](benchmark-zarr-remote.md) とまったく同じ逆転で、
   分岐点 (5 〜 25 ms) まで同じ
6. **その逆転は圧縮した転送では消せない。** deflate はワイヤを 29.5 倍縮めるのに
   100 ms で 1,353 対 1,607 と遅い。credit は論理バイトで数えるので**往復の回数が
   減らない**ためで、太い回線で効くのは in-flight を増やすことである
7. **相手の既定値で測れば、この文書の結論はほとんど嘘になる。** HSDS は既定の
   1 サービスノードだと 3.5 倍遅く見え、h5py は 100 ms で先読みしない設定だと
   4.1 倍遅く見える。**掃かずに測れば「3.5 倍勝った」と書けたところが、掃いた本当の値は
   0.83 倍 (負け)** である
8. **1 接続 (1 コア) で 16 プロセスに迫る場面がある。** `mem.h5` の RTT 0 で
   2,106 対 2,732、100 ms の `mem-gzip-noisy.h5` では 1 接続 259 が h5py の
   1 プロセス 120 の 2.2 倍・HSDS の 79 の 3.3 倍
9. **HSDS は素直に使うと落ちる。** 16 プロセスが 256 MiB ずつ頼むと、選択範囲を
   まるごとメモリに組み立てるのでサーバのプロセスがカーネルに殺される。
   16 MiB ずつに分けると落ちず、しかも 1.36 倍速い
10. **h5serv は測っていない。** HDF Group 自身が 2022 年に sunset を宣言しており、
    比較の相手として成立しない

## 条件

環境は [mdx2 の測定](benchmark-mdx2.md#測定環境) と同じ。手順は
[eval-mdx2.md](eval-mdx2.md) に従った。

| | 値 |
|---|---|
| サーバ / クライアント | `aex2-eval-1` / `aex2-eval-2` |
| REVISION | `041f341-dirty` (両方。掃引に使ったスクリプトはそのまま `5b93180` にある) |
| AEX2 サーバ | `benchmarks/mdx2/aex-decode-cache-64m.toml` (制御 50391) |
| HTTP サーバ | `benchmarks/mdx2/nginx-zarr.conf` (8080、`/mnt/aexram` を配る) |
| HSDS | `benchmarks/mdx2/hsds.sh 8 2` (5101〜5108、`/mnt/aexram/hsds` のハードリンクを読む) |
| ファイル | `mem.h5` / `mem-gzip.h5` / `mem-gzip-noisy.h5` (いずれも 4 GiB、2^30 要素の float32) |
| 読み手 | `benchmarks/mdx2/read-procs.py` (3 つとも) と `aexbench --prefault` |
| h5py / h5pyd / HSDS | 3.16.0 (同梱 libhdf5 2.0.0) / 1.0.0 / 1.0.1 |
| sysctl | 掃引の間ずっと `tuned` (`tcp_rmem` / `tcp_wmem` の上限 256 MiB) |
| iPerf3 (サーバ→クライアント) | 掃引の後に 28.3 / 143 Gbit/s (1 本 / 16 本) |

- HSDS の配り方 (8 × 2)、1 回に頼む量 (16 MiB)、h5py のキャッシュ設定は、どれも
  掃いて選んだ最良のものである。その掃引は[相手の設定を掃く](#相手の設定を掃く)にある
- **3 つとも同じバイトを読む。** HSDS はデータをコピーせず、チャンクの位置だけを
  持つ (`hsload --link` と同じ形。`benchmarks/mdx2/hslink.py` で作る)。nginx が配るのも
  AEX2 が読むのも同じ `/mnt/aexram/*.h5` で、HSDS のバケツにあるのはそこへの
  ハードリンクである
- 測る前に**3 つが同じバイトを返すことを確かめた** (`read-procs.py --check`)
- 条件は 1 回ずつ交互に回し、中央値を採った
- AEX2 のサーバはデコードキャッシュ 64 MiB のもの。**毎回ほぼ全チャンクを伸長する**

## 同じ距離 (VM 間、RTT 0.7 ms)

n = 2 の中央値、4 GiB、MiB/s。h5py の括弧内は 3 つ掃いたキャッシュ設定のうち最良のもの。
HSDS は 8 サーバ × 2 データノード、1 回 16 MiB ずつ読む。

| ファイル | h5py 1 | h5py 16 | HSDS 1 | HSDS 16 | AEX2 1 | AEX2 16 | 対 h5py | 対 HSDS | `aexbench` 16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mem.h5` (無圧縮) | 530 (none) | 2,732 (none) | 252 | 2,156 | 2,106 | **13,012** | **4.76 倍** | **6.04 倍** | 16,354 |
| `mem-gzip-noisy.h5` (27 %) | 284 (background) | 3,110 (blockcache) | 162 | 1,570 | 303 | **3,986** | **1.28 倍** | **2.54 倍** | 4,298 |
| `mem-gzip.h5` (0.9 %) | 916 (background) | 7,198 (blockcache) | 198 | 1,899 | 752 | **7,878** | **1.09 倍** | **4.15 倍** | 9,188 |

- **HSDS には 2.5 〜 6.0 倍。** 構造が同じ (サーバで読んで値を送る) 相手なので、
  これは設計の差ではなく実装の差である
- **h5py + HTTP は強い。** 圧縮がよく効くファイルでは 16 プロセスで 7,198 MiB/s 出て、
  AEX2 (7,878) に 1.09 倍まで迫る。**ワイヤに 36 MiB しか流していない**のだから当然で、
  伸長はクライアントの 16 コアが払っている
- **無圧縮のファイルでは 4.76 倍離れる。** ここは h5py が 4 GiB を range request で
  引くしかなく、1 回の読みごとに往復が要る
- AEX2 の 1 接続 (2,106) が h5py の 16 プロセス (2,732) に近いのも `mem.h5` である。
  1 接続は 1 コアなので、**16 コア分の仕事に 1 コアで並びかけている**

## 何をどこで払っているか

同じ 4 GiB についての、ワイヤのバイト数と両側の CPU。サーバ側は `/proc/net/dev` と
`/proc/<pid>/stat` の差分で、AEX2 サーバ・nginx・HSDS のノードすべてを含む。

| ファイル / 読み手 | ワイヤ | サーバ CPU | クライアント CPU | 合計 |
|---|---:|---:|---:|---:|
| `mem-gzip-noisy.h5` h5py 16 | 1,096 MiB | 0.03 s/GiB | 4.36 s/GiB | 4.39 s/GiB |
| 同 HSDS 16 | 4,103 MiB | 7.36 | 3.90 | 11.26 |
| 同 AEX2 16 | 4,102 MiB | 3.78 | **0.67** | **4.45** |
| `mem.h5` h5py 16 | 4,101 MiB | 0.07 | 2.24 | 2.31 |
| 同 HSDS 16 | 4,103 MiB | 3.71 | 4.12 | 7.83 |
| 同 AEX2 16 | 4,101 MiB | 0.67 | **0.79** | **1.46** |
| `mem-gzip.h5` h5py 16 | 175 MiB | 0.01 | 1.48 | **1.49** |
| 同 HSDS 16 | 4,104 MiB | 5.11 | 3.97 | 9.08 |
| 同 AEX2 16 | 4,102 MiB | 1.81 | 0.71 | 2.52 |

- **HSDS は同じ仕事に AEX2 の 2 〜 5 倍の CPU を使う。** サーバ側だけでなく
  クライアント側も 5 倍前後 (0.7 対 3.9 〜 4.1 s/GiB) で、**両側で高い**
- 1 コアあたりに直すと、gzip を伸長しながら送る仕事は AEX2 が 566、HSDS が 200 MiB/s
  (`mem-gzip.h5`)。**やっていることは同じ「gzip を解いて送る」なのに 2.8 倍違う**
- **h5py は合計ではいちばん安いことがある** (`mem-gzip.h5` で 1.49 s/GiB)。ワイヤに
  36 MiB しか流さず、伸長を利用者のコアでやるからである。**そのコアは利用者の
  計算に使えたはずのもの**であり、AEX2 がクライアントに残すのは 0.71 s/GiB である
- `blockcache` と `background` は**ファイルより多くのバイトを引く** (`mem.h5` で
  8,455 MiB = 2 倍)。先読みが外れたぶんで、無圧縮のファイルでは素直に損になる

## 遠くなるとどうなるか

netem で往復を足した掃引。n = 3 の中央値、16 プロセス / 16 接続 (MiB/s)。h5py は
キャッシュ 3 通りの最良値で、括弧内がその設定。

0 ms の行は[上の表](#同じ距離-vm-間rtt-07-ms)とは別の掃引なので値がずれる。
差はいちばん大きい `mem.h5` の AEX2 で 7.7 %、ほかは 3 % 以内である。

### 圧縮が効かないファイル (`mem-gzip-noisy.h5`、27 %)

| 追加 RTT | h5py 16 | HSDS 16 | AEX2 16 | 対 h5py | 対 HSDS |
|---:|---:|---:|---:|---:|---:|
| 0 ms | 3,044 (none) | 1,529 | 3,970 | 1.30 | 2.60 |
| 5 ms | 2,743 (background) | 1,543 | 3,634 | 1.32 | 2.35 |
| 25 ms | 1,992 (background) | 1,074 | 2,750 | 1.38 | 2.56 |
| 100 ms | 904 (blockcache) | 636 | 1,432 | **1.58** | **2.25** |

### 無圧縮のファイル (`mem.h5`)

| 追加 RTT | h5py 16 | HSDS 16 | AEX2 16 | 対 h5py | 対 HSDS |
|---:|---:|---:|---:|---:|---:|
| 0 ms | 2,816 (none) | 2,103 | 12,012 | 4.27 | 5.71 |
| 5 ms | 2,379 (none) | 2,089 | 10,043 | 4.22 | 4.81 |
| 25 ms | 1,642 (none) | 1,456 | 4,649 | 2.83 | 3.19 |
| 100 ms | 866 (none) | 250 | 1,669 | 1.93 | 6.68 |

### よく圧縮できるファイル (`mem-gzip.h5`、0.9 %)

| 追加 RTT | h5py 16 | HSDS 16 | AEX2 16 | 対 h5py | 対 HSDS |
|---:|---:|---:|---:|---:|---:|
| 0 ms | 6,988 (blockcache) | 1,912 | 7,653 | 1.10 | 4.00 |
| 5 ms | 6,261 (blockcache) | 1,870 | 6,563 | 1.05 | 3.51 |
| 25 ms | 4,233 (blockcache) | 1,195 | 4,043 | **0.96** | 3.39 |
| 100 ms | 1,966 (blockcache) | 628 | 1,641 | **0.83** | 2.61 |

**逆転は 5 〜 25 ms の間にある。** 理由は [Zarr のとき](benchmark-zarr-remote.md#遠くなるとどうなるか)
とまったく同じで、そこから先はリンクが律速になり、**ワイヤに 176 MiB しか流さない側**が
4,102 MiB 流す側に勝つ。圧縮が効かないファイル (1,273 対 4,102 MiB、3.2 倍) では
これが起きず、100 ms でも 1.58 倍を保つ。

**HSDS には全区間で 2.25 〜 6.68 倍。** HSDS も値を送る (ワイヤは AEX2 と同じ 4 GiB) ので
距離に対して同じ形で落ちるが、常に下にいる。100 ms の `mem.h5` で 6.68 倍と開くのは
HSDS のばらつきが大きいため (219 〜 672 MiB/s) で、この 1 点は幅で読むべきである。

### 1 プロセス / 1 接続で比べる

`mem-gzip-noisy.h5`、n = 1 (MiB/s)。h5py は 3 通りの最良値。

| 追加 RTT | h5py 1 | HSDS 1 | AEX2 1 接続 |
|---:|---:|---:|---:|
| 0 ms | 284 (background) | 162 | 303 |
| 25 ms | 265 (background) | 132 | 291 |
| 100 ms | 120 (background) | 79 | **259** |

**遠くなるほど差が開く。** AEX2 の 1 接続は 100 ms でも 259 MiB/s (RTT 0 の 85 %) を
保つのに対し、h5py は 42 %、HSDS は 49 % まで落ちる。1 本の接続に credit ぶんの
チャンクを載せ続ける形と、読むたびに往復を待つ形の差である。

**先読みがあって初めて h5py はこの値になる。** `none` だと 100 ms で 33 MiB/s ──
4 GiB を 1 MiB 前後の読みに分けて 1 回ずつ往復しているためで、`background` の
120 とは 3.6 倍違う。

### 圧縮した転送で逆転を消せるか ── 消せない

AEX2 は転送ごとに可逆の deflate を選べる (`arr.at(codec="gzip")`)。逆転が起きた
`mem-gzip.h5` でそれを試した (16 接続、n = 2、MiB/s)。

| 追加 RTT | 生バイト | `--codec gzip` | ワイヤ (生 / deflate) |
|---:|---:|---:|---|
| 25 ms | **4,020** | 2,145 | 4,102 / 139 MiB |
| 100 ms | **1,607** | 1,353 | 4,102 / 139 MiB |

**ワイヤを 29.5 倍縮めても遅くなる。** [Zarr のとき](benchmark-zarr-remote.md#deflate-はワイヤが律速のときだけ効く)
と同じ理由で、credit は**論理バイト**で数えるので圧縮しても往復の回数が減らない。
この 26 Gbit/s の回線では 100 ms でも律速は往復の回数であり、効くのは圧縮ではなく
in-flight (streams × credit × chunk) を増やすことである。**圧縮が答えになるのは
帯域が細いとき**で、そちらは [SZ3 の測定](benchmark-sz.md) と
[Zarr のリモート比較](benchmark-zarr-remote.md#帯域が細い回線では-sz3-が答えになる) にある。

### キャッシュを掃いていなかったら結論が逆になっていた

`mem-gzip.h5`、16 プロセス (MiB/s)。

| 追加 RTT | `none` | `blockcache` | `background` |
|---:|---:|---:|---:|
| 0 ms | 6,527 | **6,988** | 6,927 |
| 25 ms | 1,564 | **4,233** | 3,464 |
| 100 ms | 474 | **1,966** | 1,309 |

**100 ms では 4.1 倍違う。** 先読みしない `none` のまま測れば h5py は 474 MiB/s で、
AEX2 が 3.5 倍速いと書けてしまう。掃いた本当の値は 1,966 で、**負けているのは
こちら (0.83 倍)** である。先読みが往復を隠すぶんが、遠いところでは効く。

## 相手の設定を掃く

[Zarr のリモート比較](benchmark-zarr-remote.md#相手の設定を掃くことの重み) で
zarr-python の `async.concurrency` を掃かなければ 3.7 倍不当に遅く見えたのと同じことが、
HSDS では**もっと大きく**起きる。既定のまま測れば下の表のいちばん上の行になり、
それは HSDS ではなく**その配り方**を測っていることになる。

### HSDS はサービスノードの数で決まる

pip で入れた HSDS (`hsds --count N`) が立てるサービスノード (SN) は**1 つだけ**で、
クライアントが読むバイトは全部その 1 プロセスを通る。データノード (DN) をいくら
増やしても頭打ちになる。Docker/k8s 向けの構成では SN を複数並べてロードバランサの
後ろに置くので、ここでも**サーバを複数立てて読み手を振り分けた** (`hsds.sh`、
`read-procs.py --endpoints`)。

4 GiB、クライアント 16 プロセス、1 回 (MiB/s)。

| SN × DN | `mem-gzip-noisy.h5` | `mem.h5` |
|---|---:|---:|
| 1 × 16 (既定の形) | 374 | 416 |
| 2 × 8 | 639 | 669 |
| 4 × 4 | 1,066 | 1,288 |
| **8 × 2** | **1,317** | **1,510** |
| 16 × 1 | 1,230 | OOM で落ちた |
| 8 × 4 | 1,231 | OOM で落ちた |
| 16 × 2 | 1,150 | OOM で落ちた |

**SN を 1 つから 8 つにすると 3.5 倍になる。** DN の数はほとんど効かない ──
律速は読み出しでも伸長でもなく、**値をクライアントに送り出す 1 プロセス**である。
以降はすべて 8 × 2 で測った。

### 1 回に頼む量

HSDS は**選択範囲をまるごとメモリに組み立ててから**答える。16 プロセスが 256 MiB ずつ
一度に頼むと、サーバのメモリが尽きて**カーネルにプロセスを殺され**、読み手は
`IncompleteRead` で落ちる。h5pyd の側で分けて頼めばよい (`read-procs.py --piece`)。

`mem.h5`、8 × 2、16 プロセス (MiB/s)。

| 1 回の大きさ | 1 MiB | 4 MiB | **16 MiB** | 64 MiB | 128 MiB | 256 MiB (分けない) |
|---|---:|---:|---:|---:|---:|---:|
| `mem.h5` | 1,337 | 1,653 | **2,053** | 1,891 | OOM | OOM (生き残った回は 1,510) |

**分けたほうが速い。** 16 MiB で 2,053 MiB/s ── 分けずに頼んで運よく生き延びた回
(1,510) の 1.36 倍である。以降はすべて 16 MiB で測った。

分けているのは h5pyd だけで、h5py と AEX2 には**スライス全体を 1 回で**読ませている。
どちらも要求を流しながら返すので、分けても得がない。

### h5py のファイルオブジェクトのキャッシュ

fsspec のキャッシュ方式は**ファイルによって最良が変わる**ので、3 つとも掃いた
(ブロック長は 1 / 4 / 16 MiB を試して 4 MiB が最良だったため、それで固定)。

16 プロセス、RTT 0、n = 2 の中央値 (MiB/s)。

| ファイル | `none` | `blockcache` | `background` |
|---|---:|---:|---:|
| `mem.h5` | **2,732** | 2,389 | 1,872 |
| `mem-gzip-noisy.h5` | 3,108 | **3,110** | 2,988 |
| `mem-gzip.h5` | 6,947 | **7,198** | 6,900 |

差は 1.5 倍まで開く。**無圧縮のファイルでは先読みが損になる** ── `background` は
8,455 MiB、つまりファイルの 2 倍を引いていた。

## この回の限界

- **iPerf3 を掃引の前に測っていない。** 後の値 (28.3 / 143 Gbit/s) は
  [これまでの測定](benchmark-hdf5.md#条件) の範囲 (26 〜 37 / 132 〜 170) の内側なので
  帯域が動いたとは見ていないが、**手順どおりではない**
- **RTT 0 の表は n = 2** で、距離の掃引は n = 3。`mem.h5` の AEX2 1 接続 (1,721 〜 2,492)
  と h5py 16 プロセス (2,517 〜 2,953) は振れが大きい
- **距離の掃引は 16 並列だけ**で回した。1 プロセスの値は `mem-gzip-noisy.h5` の
  n = 1 しかない
- **100 ms の HSDS は振れが大きい** (`mem.h5` で 219 〜 672)。6.68 倍という比はこの
  ばらつきを含んでいる
- **HSDS の配り方は 8 × 2 で止めた。** メモリが足りず 8 × 4 や 16 × 2 は
  無圧縮のファイルで落ちる。**この VM のメモリ (31 GB のうち 17 GB が tmpfs) が
  効いている**可能性があり、大きなメモリなら別の最適点があり得る
- **h5pyd の 1 回の大きさ (16 MiB) は `mem.h5` でしか掃いていない**
- **h5py の ros3 ドライバは使えなかった。** ホイールの libhdf5 が ROS3 無効で
  ビルドされているためで、S3 越しに読む形は測っていない。fsspec のファイル
  オブジェクトで代替した
- **kerchunk は測っていない。** HDF5 のチャンクを Zarr に見せて obstore で並列に
  引く形は、ここでの h5py + HTTP よりさらに強い可能性がある
- HSDS の**書き込み性能は見ていない**。この比較は読みだけである

## 再現方法

nginx・HSDS・AEX2 サーバを立て、ファイルを HSDS にリンクしてから、クライアントの
tmux で掃引を回す。

```console
$ ssh aex2-eval1 'sudo nginx -c /home/mdxuser/aex2/benchmarks/mdx2/nginx-zarr.conf'
$ ssh aex2-eval1 "bash -lc 'cd aex2 && bash benchmarks/mdx2/hsds.sh 8 2 &&
    export HS_ENDPOINT=http://localhost:5101 HS_USERNAME=test HS_PASSWORD=test HS_BUCKET=hsds &&
    for f in mem.h5 mem-gzip.h5 mem-gzip-noisy.h5; do
      ~/hsds-venv/bin/python benchmarks/mdx2/hslink.py \$f /home/test/\$f 1048576; done'"
```

3 つが同じバイトを返すことは、測る前に確かめる。

```console
$ ssh aex2-eval2 "bash -lc 'cd ~/aex2 && export HS_USERNAME=test HS_PASSWORD=test HS_BUCKET=hsds &&
    S=192.168.100.207 && for f in mem.h5 mem-gzip.h5 mem-gzip-noisy.h5; do
      .venv/bin/python benchmarks/mdx2/read-procs.py http://\$S:8080/\$f --backend h5py \
        --check aex:http://\$S:50391/\$f
      .venv/bin/python benchmarks/mdx2/read-procs.py http://\$S:5101/home/test/\$f --backend h5pyd \
        --check aex:http://\$S:50391/\$f; done'"
```

掃引は 3 本に分けて回した。1 本目が RTT 0 の表と CPU・ワイヤの内訳、2 本目が距離、
3 本目が 1 プロセスの距離である。

```console
$ ssh aex2-eval2 "tmux new-session -d -s h5 'cd aex2 &&
    SERVER=192.168.100.207 CLIENT=192.168.101.235 RTTS=0 COUNTERS=1 \
    FILES=\"mem-gzip-noisy.h5 mem.h5 mem-gzip.h5\" \
    bash benchmarks/mdx2/h5-remote-sweep.sh 2 2>&1 | tee ~/h5-remote-a.txt'"
$ ssh aex2-eval2 "tmux new-session -d -s h5b 'cd aex2 &&
    SERVER=192.168.100.207 CLIENT=192.168.101.235 RTTS=\"0 5 25 100\" PROCS=16 COUNTERS=1 \
    FILES=\"mem-gzip-noisy.h5 mem.h5 mem-gzip.h5\" \
    bash benchmarks/mdx2/h5-remote-sweep.sh 3 2>&1 | tee ~/h5-remote-b.txt'"
$ ssh aex2-eval2 "tmux new-session -d -s h5c 'cd aex2 &&
    SERVER=192.168.100.207 CLIENT=192.168.101.235 RTTS=\"25 100\" PROCS=1 COUNTERS=1 \
    FILES=mem-gzip-noisy.h5 bash benchmarks/mdx2/h5-remote-sweep.sh 1 2>&1 | tee ~/h5-remote-c.txt'"
```

**HSDS は掃引の前に立て直し、8 つのポートが全部応答することを確かめてから回す。**
1 つでも死んでいると、そこに割り当てられた読み手は**エラーにならずに止まる**ので、
掃引が固まったまま進まない。

```console
$ ssh aex2-eval1 'ss -ltn | grep -oE ":51[0-9]+" | sort -u'
```

配り方と 1 回の大きさの掃引は、サーバを立て直しながら `--endpoints` と `--piece` を
振って測った。

```console
$ ssh aex2-eval1 "bash -lc 'cd aex2 && bash benchmarks/mdx2/hsds.sh stop; sleep 3;
    bash benchmarks/mdx2/hsds.sh 4 4'"
$ ssh aex2-eval2 "bash -lc 'cd aex2 && HS_USERNAME=test HS_PASSWORD=test HS_BUCKET=hsds \
    .venv/bin/python benchmarks/mdx2/read-procs.py \
      http://192.168.100.207:5101/home/test/mem.h5 --backend h5pyd \
      --endpoints 4 --piece 4194304 --procs 16 --reps 2'"
```
