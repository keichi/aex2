# mdx2 での評価手順

mdx2 上の VM 2 台で測るときの決まりごと。測定結果そのものは `docs/benchmark-*.md`
に書き、ここには**毎回同じにすべき手順**だけを置く。

## 構成

| | サーバ | クライアント |
| ホスト名 (記録に書く名前) | `aex2-eval-1` | `aex2-eval-2` |
| SSH の別名 (コマンドで使う名前) | `aex2-eval1` | `aex2-eval2` |
|---|---|---|
| アドレス | 192.168.100.207/23 | 192.168.101.235/23 |
| CPU / メモリ | 16 vCPU / 31 GiB | 同じ |
| OS | Ubuntu 24.04.4 (kernel 6.8.0-136) | 同じ |
| データ | `/mnt/aexram` (tmpfs 12 GiB)、`~/disk` (virtio) | なし |

- SSH の別名は `~/.ssh/config` にあり、`mdx2-gw` 経由、ユーザ `mdxuser`
- VM 間は RTT 約 0.7 ms、MTU 1442 (MSS 1390)。帯域は共有で、**同条件でも日によって振れる**
- ハードウェアの詳細は [benchmark-mdx2.md](benchmark-mdx2.md#測定環境) の「測定環境」

## 原則

1. **VM 上で手編集しない。** コードは Mac で編集し、`benchmarks/mdx2/sync.sh` で送る
2. **docs に載せる数字は、`~/aex2/REVISION` が `-dirty` でないときに測る。**
   未追跡のファイルがあっても `-dirty` になる。
   試行錯誤は dirty でよいが、記録用には一度コミットしてから送り直す
3. **サーバ設定はリポジトリの TOML を使う。** ホームに置いた一時的な TOML で測った
   結果を記録するなら、その TOML も `benchmarks/mdx2/` にコミットする
4. **フィクスチャは決まった場所に決まった名前で置く** (下記)。場所を増やしたらこの文書に足す

## 初回セットアップ (両方の VM)

作り直したときや、新しい VM を足したときだけ行う。

```console
$ sudo apt-get install -y build-essential protobuf-compiler iperf3 linux-tools-generic
$ curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.98.1
$ rustup component add clippy rustfmt
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```

`~/.bashrc` の `. "$HOME/.cargo/env"` は非対話シェルの早期 return より後にあるので、
`ssh host cargo ...` では cargo が見つからない。**リモートでコマンドを打つときは
`bash -lc '...'` で包む** (`.profile` が PATH を通す)。

Python 環境はリポジトリのコピーを送ったあとに作る (`.venv` は同期で消えない)。

```console
$ cd ~/aex2 && uv venv --python 3.13 && uv pip install maturin numpy pytest
```

v1 との比較をするときだけ、`~/aex` に v1 (`4dcb6c3`) を置いて `uv sync` する。

### ツールのバージョン

| | 現在の値 | 合わせる相手 |
|---|---|---|
| Rust | 1.98.1 | Mac と同じ。上げるときは両方同時に |
| Python | 3.13.15 (uv 管理) | CI の新しい方 |
| numpy | 2.5.3 | — |
| protoc | 3.21.12 (apt) | — |

測定結果にはこの表の値を「条件」として書く。

## データの置き場所

すべてサーバ (`aex2-eval1`) 側。サーバ設定の `roots` はこの 2 か所。

| パス | 中身 | 作り方 |
|---|---|---|
| `/mnt/aexram/mem.npy` | 4 GiB (2^30 要素)、float32、メモリ常駐 | `target/release/mknpy /mnt/aexram/mem.npy 1073741824` |
| `/mnt/aexram/m3/` | Python ベンチ用の 10M / 100M / 1G | `.venv/bin/python benchmarks/create_benchmark_data.py --output-dir /mnt/aexram/m3 --sizes 10M 100M 1G --dtypes float32` |
| `~/disk/disk.npy` | 4 GiB (2^30 要素)、float32、virtio ディスク | `target/release/mknpy ~/disk/disk.npy 1073741824` |

- `mknpy` の第 2 引数はバイト数ではなく**要素数**
- 揃っているかは `ssh aex2-eval1 'ls -l /mnt/aexram /mnt/aexram/m3 ~/disk'` で確かめる。
  `/mnt/aexram` が空なら再起動で消えている
- **`/dev/shm` は使わない。** systemd がログアウト時に中身を消す
- `/mnt/aexram` は **fstab に無いので再起動で消える**。再起動したら張り直して上の
  フィクスチャを作り直す。残すなら `/etc/fstab` に
  `tmpfs /mnt/aexram tmpfs size=12G,mode=1777 0 0` を足す

  ```console
  $ sudo mkdir -p /mnt/aexram && sudo mount -t tmpfs -o size=12G,mode=1777 tmpfs /mnt/aexram
  ```

- コールドなディスク読みを測るときは、各回の前に
  `target/release/dropcache ~/disk/disk.npy` でそのファイルのページだけ落とす (root 不要)
- 生成したフィクスチャはリポジトリの外にあるので同期では消えない。`~/aex2` の中には置かない

## コードの転送とビルド

```console
$ benchmarks/mdx2/sync.sh                 # 両方
$ benchmarks/mdx2/sync.sh aex2-eval1      # 片方だけ
```

作業ツリーを `~/aex2` へ rsync し、`git describe --always --dirty` を
`~/aex2/REVISION` に書き、両方で `cargo build --release` と
`maturin develop --release` を並列に走らせる。

- `.gitignore` にあるもの (`target/`、`.venv/`、`*.so`、`*.npy`) は送らず、VM 側のものも消さない。
  Mac の `_aex.abi3.so` が VM に渡ると `import aex` が `invalid ELF header` で落ちる
- git を使わないのは、コミット前の変更を Linux で試すため (`tcp.congestion` など
  Linux でしか動かない経路は、CI より先にここで試す)
- zsh で手で rsync するなら `"${h}:aex2/"` と書く。`$h:aex2/` は `:a` が修飾子として
  展開され、**ローカルのリポジトリへ**同期してしまう
- macOS の `rsync` は openrsync で、`--filter=':- .gitignore'` と `--delete` を組み合わせると
  `.venv` まで消す。`--exclude-from` を使う

テストも VM で回せる。

```console
$ ssh aex2-eval1 "bash -lc 'cd aex2 && cargo clippy --all-targets --all-features -- -D warnings && cargo test --release'"
```

## サーバの起動と停止

設定ごとに TOML を分け、ポートも分けて並べて立てる (100 ずつずらす)。

| TOML | 制御 / データ | 違い |
|---|---|---|
| `aex.toml` | 50191 / 50192 | 基準 (`read_buffers` は既定の 3) |
| `aex-read-buffers-1.toml` | 50291 / 50292 | `read_buffers = 1` |

```console
$ for c in aex aex-read-buffers-1; do
    ssh aex2-eval1 "cd aex2; mkdir -p ~/logs; setsid nohup target/release/aex-server \
      --config benchmarks/mdx2/$c.toml > ~/logs/$c.log 2>&1 < /dev/null &"
  done
$ sleep 1; ssh aex2-eval1 'ss -ltn | grep -E ":50[0-9]9[12]"'   # 4 行出れば立ち上がっている
$ ssh aex2-eval1 'pkill -x aex-server'                    # 止める
```

- `setsid nohup ... < /dev/null &` を**単独の文**にしないと ssh が返ってこない。
  `cd aex2 && setsid ... &` と書くと `&&` の連なり全体が ssh の出力を掴んだまま裏に回る
- ログは `~/logs/<TOML 名>.log`。一時的にポートだけ変えるなら `--control-addr` /
  `--data-addr` で上書きできる
- `data_advertise_host` は空のままでよい。制御プレーンに届いたアドレスがそのまま使われる

## 測定の手順

1. **前片付けと状態確認** (両方の VM)

   ```console
   $ for h in aex2-eval1 aex2-eval2; do ssh $h 'hostname; pgrep -a "aex|iperf3"; uptime;
       cat aex2/REVISION; rustc -V 2>/dev/null || ~/.cargo/bin/rustc -V; iperf3 -v | head -1;
       sysctl net.core.wmem_max net.ipv4.tcp_congestion_control'; done
   ```

   `pgrep` は何も出さないこと。load average は 1 未満。sysctl の既定は
   `net.core.wmem_max = 212992`、`net.ipv4.tcp_congestion_control = cubic`。
   変えて測ったら、**終わったら戻す**
2. **同期**: `benchmarks/mdx2/sync.sh`。記録用なら REVISION が `-dirty` でないこと
3. **ネットワークの基準値**: AEX2 を測る**直前と直後**に iPerf3 を測り、両方を記録する。
   向きは AEX2 と同じ**サーバ → クライアント** (`-R`)。1 本と 16 本の両方を取る

   ```console
   $ ssh aex2-eval1 'iperf3 -s -D'; sleep 1
   $ ssh aex2-eval2 'iperf3 -c 192.168.100.207 -t 10 -Z -R' | grep receiver
   $ ssh aex2-eval2 'iperf3 -c 192.168.100.207 -t 10 -Z -R -P 16' | grep 'SUM.*receiver'
   $ ssh aex2-eval1 'pkill -x iperf3'
   ```

   値は最後の `receiver` 行 (16 本なら `[SUM]` の行)。Gbit/s は 10^9 ビットなので、
   MiB/s にするには × 119.2。

   **帯域は同じ日のうちでも 25 % 前後動く** (2026-09-17 の 1 本は 35.5 と 26.3 Gbit/s)。
   だから数字を比べてよいのは、**同じ回のうちに交互に測った条件どうし**だけ。
   過去の記録と比べたいときは、その記録の条件も同じ回に測り直す。直前と直後の
   iPerf3 が 10 % 以上違ったら、その回は帯域が動いていたものとして測り直す。
   `benchmark-mdx2.md` と `benchmark-m4.md` の iPerf3 の値は向きが記録されていない
4. **サーバ起動** (上記)
5. **測る**: 条件は**交互に 1 回ずつ**回す (`--reps 1` を外側のループで n 回)。
   片方をまとめて測ってからもう片方、はしない。この VM は同条件でも振れが大きい

   ```console
   $ for r in 1 2 3 4 5; do for s in 1 16; do for p in 50191 50291; do
       ssh aex2-eval2 "bash -lc 'cd aex2 && target/release/aexbench \
         http://192.168.100.207:$p mem.npy --bytes \$((4<<30)) \
         --streams $s --reps 1 --label p$p'"
     done; done; done | tee ~/aex2-bench.txt
   $ sed -E 's/ +median +([0-9]+).*/ \1/' ~/aex2-bench.txt | sort -k1,6 -k7n
   ```

   並べ替えると条件ごとに n 行ずつ並ぶので、真ん中の行が中央値。`--label` を付けないと
   どのサーバの行か分からない。
   チャンクは既定 (サーバ推奨値) で測る。過去の記録と比べるときだけ、その記録の
   `--chunk` / `--credit` に合わせる (`benchmark-m4.md` は `--chunk $((16<<20))`)
6. **直後の iPerf3** (手順 3 と同じ)
7. **後片付け**: サーバと iperf3 を止め、手順 1 のコマンドで何も残っていないことと
   sysctl が既定値であることを確かめる

## 結果の記録

`docs/benchmark-<テーマ>.md` に書き、README の一覧に足す。既存の文書と同じく次の節を持つ。

- **条件**: 測定日、ホスト名、`REVISION`、サーバ設定ファイル、手順 1 で出したバージョン、
  直前と直後の iPerf3 の値 (向きも)、sysctl を変えたならその値
- **結果**: 表。単位は MiB/s。中央値と回数 (n) を明記する
- **再現方法**: 実際に打ったコマンド。この文書の手順と違うところだけでもよい
