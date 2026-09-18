# io_uring によるデータプレーンの再構成 — 設計と検証計画

**状態: 提案 (未実装)。** 作成 2026-09-18。この文書は実装と測定の指示書であり、
測定結果は別途 `docs/benchmark-uring.md` に記録する。

## 1. 背景

これまでの測定 ([Linux](benchmark-linux.md)、[mdx2](benchmark-mdx2.md)、
[共有読みプール](benchmark-read-pool.md)) から分かっていることは 4 つある。

1. **メモリ常駐・低遅延では、接続 1 本の律速はサーバの接続スレッドの CPU である。**
   mdx2 の VM 間 (tmpfs、逐次) で送りスレッドは 98 % で飽和し、その内訳は
   `clear_page_erms` (送信ページの確保とゼロ埋め) 30.5 %、`_copy_to_iter`
   (pread のコピー) 25.9 %、`_copy_from_iter` (writev のコピー) 6.7 %、
   virtio のキックと skb の会計などが約 13 %
2. **スレッド間でバッファを受け渡すと負ける。** VM ではダブルバッファリングが
   速度 −3 %・CPU +29 %、共有読みプールは 16 接続で逐次の半分。原因はスレッド数
   ではなく受け渡しそのもの
3. **コールドディスクでは、スループットは同時に走る pread の数で決まる。**
   読みと送りを重ねる価値があるのはこの場合だけ
4. **受信側との余裕は小さい。** サーバ (逐次) 0.35 s/GiB に対し、クライアントは
   0.29〜0.31 s/GiB。サーバだけを軽くしても、接続 1 本あたりの伸びは 15〜20 % 程度で
   受信側が律速になる見込み

いまの構成では「逐次の局所性 (読んだスレッドがそのまま送る)」と「pair の重なり
(ディスク I/O が飛んでいる間に送る)」が両立しない。io_uring を**接続を処理する
スレッド自身が投入し、自身が刈り取る**形で使えば両立できる、というのがこの計画の
仮説である。ページキャッシュにヒットした読みは投入時にそのスレッド上で完了し、
ミスした読みは飛んでいる間に送信が進む。

## 2. 目標と非目標

目標:

- コールドな読みと送信を、スレッド間の受け渡しなしに重ねる
- ゼロコピー送信 (`SEND_ZC`) で送信側の CPU を削る (カーネル 6.0 以降)
- `read_buffers` の自動判定を、先読み深さという 1 つのノブに置き換える

非目標:

- **プロトコルとクライアントは変えない。** ワイヤ形式、FETCH / DATA の意味、
  クライアントの実装はそのまま
- **データプレーンに async ランタイムを持ち込まない。** tokio はコントロールプレーンに
  閉じたまま。io_uring は `io-uring` クレート (tokio-rs/io-uring) を直接使う。
  tokio-uring / monoio / glommio は使わない
- `sendfile` / splice は扱わない ([sendfile.md](sendfile.md) の判断を維持)。
  無加工の連続範囲に限った splice 経路は、この計画の完了後に別途検討する
- O_DIRECT は扱わない (§8 の未決事項)
- macOS の経路は変えない

## 3. 現状の構成 (変更前)

- コントロールプレーンは tokio + tonic。データプレーンは `dataplane.rs` で、
  `aex-data-accept` スレッドが 20 ms 間隔のポーリングで accept し、接続ごとに
  `aex-data-conn` スレッドを 1 本立てる
- 接続スレッドは FETCH を受信し、ticket を検証し、範囲を `read_buffer_bytes`
  (512 KiB) の piece に切って、piece ごとに `read_range` (pread) → `write_frame`
  (`writev` でヘッダ 32 B とペイロード) を逐次に行う
- `read_buffers ≥ 2` のときだけ `aex-data-read` スレッドが付き、work / pieces / free の
  3 本の channel でバッファを循環させる (`reader.rs`)
- 停止は、ソケットの 20 ms 読みタイムアウトで `stop` フラグを見て検知する

**要確認:** `config.rs` の `Transfer::default()` は `read_buffers: 1` だが、README の
チューニング節は「既定の `0` (自動)」と書いており、`dataplane.rs` は
`transfer.buffers_for(granted_streams)` を呼んでいる。`buffers_for` の定義が
`config.rs` に見当たらない。作業開始前に、ビルドが通るか、どれが正しいかを確かめること。

## 4. 設計

### 4.1 全体像

2 段階で入れる。

| | 段階 1 | 段階 2 |
|---|---|---|
| ring に載せるもの | piece の読み (`READ`) | 読み + FETCH の受信 (`RECV`) + DATA の送信 (`SEND_ZC` / `SEND`) |
| スレッド | 今と同じ 1 接続 1 スレッド | コア数ぶんの worker、接続を割り当て |
| ソケット I/O | 今と同じ blocking | ring 上で非同期 |
| 効くこと | コールドでの重なり | 加えて、ゼロコピー送信、FETCH をまたいだ先読み、停止ポーリングの廃止、スレッド数の固定 |
| カーネル | 5.9 以降 (非同期バッファ読み) | 5.15 でも動く。ゼロコピー送信は 6.0 以降 |

段階 1 のうちから、接続の処理を「接続の状態 + それを進める関数」の形に分けて書く。
段階 2 では「1 スレッドに 1 接続の状態」を「1 スレッドに複数の状態」にするだけで
済むようにするためである。

### 4.2 バックエンドとの境界

`ArrayDataset` に任意実装のメソッドを足す。既定は `None`。

```rust
/// File extents backing `[offset, offset + len)` of the logical stream, for
/// backends that store the selection uncompressed. `None` means the caller
/// must use `read_range`.
fn extents(&self, layout: &SelectionLayout, offset: u64, len: u64) -> Option<Extents>;
```

- `Extents` は「fd・ファイル上のオフセット・長さ・出力バッファ上の位置」の列。
  非連続選択では、今 `read_with` が内部で行っている「近接断片をまとめて読む
  (隙間 4 KiB 以下、1 回 1 MiB まで) → 要素単位で切り出す」を、「まとめた断片の
  READ を並べて投入」と「全部完了したら切り出す」に分けられるよう、切り出しに必要な
  情報も返す
- npy は実装する。HDF5 は非圧縮・連続格納のデータセットだけ実装し、圧縮チャンクは
  `None` (デコードキャッシュ経由の `read_range`)
- SPEC §7.1 の `zero_copy_source()` と同じ発想で、トレイトの既存部分は変えない

### 4.3 段階 1: 読みだけ ring

`ReadPipeline` の inline モード (`read_buffers = 1`) の実装を置き換える。インタフェース
(`request` / `next_piece` / `recycle`) は保ち、`dataplane.rs` の変更を最小にする。

- 接続スレッドごとに ring を 1 本 (エントリ数 64 程度)。生成フラグは
  `SINGLE_ISSUER | DEFER_TASKRUN` を試し、`EINVAL` なら外して作り直す
- FETCH を受けたら、枠 (`read_depth` 枚、既定 4) に空きがある限り piece の READ を
  投入する。1 piece は 1 個以上の READ (断片の数)
- piece の全 READ が完了したら送信待ち。送信は **piece の offset 順**に今の
  `write_frame` で行い、送ったバッファを枠に返す
- READ の short read は残りを再投入する。0 バイト (EOF) はファイルが縮んだとして
  今と同じエラーにする
- 読みの失敗はその FETCH を ERROR で返す。**投入済みの READ の完了をすべて刈り取って
  からバッファを返す** (SQE がまだバッファを参照しているため)
- `extents` が `None` のバックエンドは、今と同じく接続スレッド上で `read_range` を
  同期に呼ぶ
- バッファは登録しない (普通の `READ`)。読みだけなら fixed buffers の利得は小さく、
  memlock を消費するため

### 4.4 段階 2: ソケットも ring、worker モデル

#### スレッドと割り当て

- worker スレッドを `data_workers` 本 (既定は `available_parallelism()`) 立て、各 worker を
  コアに固定する (Linux のみ。失敗しても続行)。各 worker は ring 1 本と登録バッファの
  プール 1 つを持つ
- コントロールプレーンの tokio ランタイムは `worker_threads` を 2 程度に絞る
- accept は今の専用スレッドのままでよい。ハンドシェイク前の fd を、**接続数が最も
  少ない worker** に channel で渡し、その worker の eventfd を叩いて起こす。同じ
  セッションの接続はなるべく別の worker に置く (ハンドシェイク後にセッションが分かる
  ので、ハンドシェイクだけは accept スレッドで行ってもよい)
- 割り当ては固定で、worker 間の移動はしない。偏りはクライアントのワークスティーリング
  が吸収する

#### ring に投入する操作

| 操作 | いつ投入するか | 完了で何をするか |
|---|---|---|
| `RECV` | 接続ごとに常に 1 つ | FETCH (ヘッダ 32 B + ticket 16 B) を組み立て、揃ったら検証して piece を待ち行列へ。部分受信は続きを再投入 |
| `READ` / `READ_FIXED` | 接続の枠に空きがあり、待ち行列に piece がある | piece の未完了断片数を減らし、0 なら送信待ちへ |
| `SEND_ZC` (6.0 以降) / `SEND` | 接続に送信中のものがなく、先頭の piece が送信待ち | 部分送信なら残りを再投入。完了なら次の piece へ |
| 解放通知 (`SEND_ZC` の 2 つ目の CQE) | — | バッファを worker のプールへ返す |
| eventfd の `READ` | worker ごとに常に 1 つ | 新しい接続の受け取り、または停止 |

`user_data` の 64 ビットに「種別・接続番号・バッファ番号」を詰めて完了を振り分ける。
ループは「`submit_and_wait` → CQE を種別ごとに処理 → 全接続をラウンドロビンで回して
次の SQE を積む」の繰り返し。

#### 順序の約束 (正しさに関わる)

- **1 接続につき、`RECV` と送信 SQE はそれぞれ同時に 1 つまで。** 同じソケットに独立した
  送信 SQE を複数並べると、再試行の都合で順序が入れ替わりうる
- **送信は FETCH の到着順、FETCH 内は piece の offset 順。** クライアントの
  `receive_data` はフレームの offset で位置を決めるので FETCH 内の入れ替えは受け付ける
  ように見えるが、FETCH をまたいだ入れ替えは受け付けない前提で作る (§8 で要確認)
- READ はいくつ並べてもよい (完了順は不定)

#### FETCH をまたいだ先読み

`RECV` が常駐しているので、credit で後続の FETCH が届いていれば、今の FETCH を送って
いる間に次の FETCH の READ を投入してよい。枠の上限は接続単位で共通。

#### バッファ

- 登録バッファは「ページ境界の手前 32 B にヘッダ、ページ境界からペイロード
  `read_buffer_bytes`」の配置で確保し、ヘッダとペイロードを 1 回の送信で送る
  (ペイロードがページ境界に揃うので、将来 O_DIRECT を入れても困らない)
- プールは worker 単位。接続には枠 (同時に持てる枚数の上限) だけを持たせる
- 枠の既定: ゼロコピー送信なら `read_depth + ceil(sndbuf / read_buffer_bytes)`、
  コピー送信なら `read_depth + 1`。`sndbuf` は `tcp.sndbuf` が 0 なら
  `net.ipv4.tcp_wmem` の上限を読む
- ゼロコピー送信では、バッファは**解放通知まで**返さない。コピー送信 (`SEND`) では
  送信の CQE で返す。両者の違いはこの 1 点に閉じ込める
- memlock の必要量は「worker 数 × プール」。不足で登録に失敗したら、登録なしの
  `READ` + コピー送信に落として警告を出す

#### 後始末・停止・タイムアウト

- 接続の終了: 未完了の SQE を `ASYNC_CANCEL` で取り消し、**すべての CQE と解放通知が
  返るのを待ってから** fd を閉じ、枠を返す
- アイドルタイムアウト: 接続ごとに最後にフレームが届いた時刻を持ち、ループの待ちに
  タイムアウト (`submit_with_args`) を付けて見回る
- サーバの停止: 各 worker の eventfd を叩き、全接続に上の終了処理をかけて join
- ハンドシェイクのタイムアウト (10 s) も同じ見回りで扱う

#### エラーと例外

- 読みの失敗は ERROR フレームで返す。ERROR は接続ごとの小さな固定領域から通常の
  `SEND` で送る
- ゼロコピー送信が `ENOBUFS` を返したら、そのバッファは通常の `SEND` で送り直す
- 解放通知が「コピーした」と報告したら数える (カーネルが対応していれば
  `IORING_SEND_ZC_REPORT_USAGE` を使う。対応版は要確認)。ループバックでは常にコピーに
  なる
- `extents` が `None` の経路は、worker 上で同期に読まない。小さなブロッキング用
  スレッドプールで `read_range` を行い、完了を eventfd で worker に知らせる

### 4.5 カーネル機能の検出とフォールバック

起動時に ring を作り、`IORING_REGISTER_PROBE` で対応オペコードを調べて、次のどれで
動くかを決めてログに出す。

| モード | 条件 | 送信 |
|---|---|---|
| `uring-zc` | `SEND_ZC` あり (6.0 以降) | ゼロコピー |
| `uring-copy` | io_uring は使えるが `SEND_ZC` なし (5.15 など) | コピー |
| `legacy` | io_uring が使えない (macOS、seccomp、`kernel.io_uring_disabled`) | 今の経路 |

段階 1 の時点では `uring-read` / `legacy` の 2 つ。

必要な機能の版:

| 機能 | 版 | OCTOPUS (5.15) |
|---|---|---|
| `READ` / `READ_FIXED`、登録バッファ | 5.1〜5.6 | 可 |
| `RECV` / `SEND`、ソケット操作のポーリング駆動 | 5.6〜5.7 | 可 |
| 非同期バッファ読み (ext4 / XFS など) | 5.9 | 可 |
| `ASYNC_CANCEL`、タイムアウト付きの待ち | 5.5〜5.11 | 可 |
| io-wq ワーカーの CPU 固定 | 5.14 | 可 |
| `SEND_ZC`、`SINGLE_ISSUER` | 6.0 | 不可 |
| `DEFER_TASKRUN` | 6.1 | 不可 (最適化のみ) |

### 4.6 設定項目

`[server.transfer]` に足す。既定値は測定で決める。

| 項目 | 既定 (仮) | 意味 |
|---|---|---|
| `io_engine` | `auto` | `auto` / `uring-zc` / `uring-copy` / `legacy`。`auto` は §4.5 の検出に従う |
| `read_depth` | 4 | 接続あたりの先読み piece 数 |
| `data_workers` | 0 (= コア数) | 段階 2 の worker 数 |

ベンチマーク用に環境変数 `AEX_IO_ENGINE`、`AEX_READ_DEPTH` でも上書きできるようにする。
`read_buffers` は `legacy` のときだけ意味を持つ。

## 5. 検証計画

### 5.1 手順 0: poolbench で効果を先に確かめる

サーバを改修する前に、`benchmarks/rust/src/bin/poolbench.rs` に次を足す。
既存の `serial` / `pair` / `pool` はそのまま残す。

| モード | 内容 | 相当 |
|---|---|---|
| `uring` | 接続スレッドごとに ring。READ を `--depth` 個先行投入、送信は blocking `writev` | 段階 1 |
| `zc` | `serial` の送信を `MSG_ZEROCOPY` にする (エラーキューで完了を刈る) | 送信側の CPU 削減だけを見る |
| `uring-zc` | READ も送信 (`SEND_ZC`) も ring。`--workers` で接続を worker に割り当て | 段階 2 |
| `uring-copy` | `uring-zc` の送信を `SEND` にしたもの | 段階 2 の 5.15 相当 |

あわせて次を足す。

- `sink` に、受信した総バイト数と**受信側の CPU (s/GiB)** の報告。受信側が律速に
  なったかを判定するため
- `send` に、ゼロコピーが「コピー」に戻った割合と `ENOBUFS` の回数の報告
- 正しさの確認用に `--verify`: 送信側がファイルの各 piece のチェックサムを、受信側が
  受け取ったデータのチェックサムを出し、一致を確かめる (性能測定では外す)

poolbench は FETCH の往復を含まず、各接続が自分の範囲を押し出すだけなので、
**FETCH をまたいだ先読みの効果はここでは測れない**。それは手順 2 で `aexbench` を使う。

### 5.2 測定条件

環境:

- **mdx2 の VM 2 台** (kernel 6.8、virtio NIC、[eval-mdx2.md](eval-mdx2.md) の手順)。主戦場
- **Linux 機 (5900X)**: ループバックなのでゼロコピーは常にコピーに戻る。`uring` の
  重なりの確認と、ゼロコピーが**遅くなる**ことの確認にだけ使う
- **OCTOPUS (kernel 5.15)**: `uring` と `uring-copy` のみ。データを置くファイルシステムと
  ネットワーク経路 (Ethernet / IPoIB) を記録する

データ: tmpfs (メモリ常駐)、ディスク (`--cold`、`posix_fadvise(DONTNEED)`)。

掃引: 接続数 1 / 4 / 16、`--depth` 1 / 2 / 4 / 8、`--workers` (段階 2) は 接続数と同じ・
コア数・コア数の半分。

記録する量: スループット (MiB/s)、送信側 CPU (s/GiB)、受信側 CPU (s/GiB)、
コピーに戻った割合、`ENOBUFS` の回数、`perf` の送信スレッドの上位関数 (代表条件のみ)、
io-wq スレッドの CPU 時間 (ネットワーク FS の場合)。

VM は揺れが大きいので、比較する 2 方式は交互に測る ([mdx2 の測定](benchmark-mdx2.md) の
手法)。値は中央値、n ≥ 5。

### 5.3 判定基準

| 問い | 合格とみなす条件 |
|---|---|
| 段階 1 は入れる価値があるか | tmpfs で `uring` が `serial` の ±3 % 以内、かつコールドで `pair` 以上 |
| ゼロコピー送信は効くか | mdx2 で `zc` または `uring-zc` の送信側 CPU が `serial` より 25 % 以上少ない |
| 接続 1 本あたりの速度は伸びるか | 送信側 s/GiB が受信側を下回ったら、以後は受信側の問題として扱う |
| worker モデルは損をしないか | 接続数 ≤ worker 数で `uring-zc` が 1 接続 1 スレッドの版と同等 |
| 5.15 で段階 2 を入れる価値があるか | OCTOPUS で `uring-copy` が `uring` (段階 1) より明確に速い。そうでなければ OCTOPUS では段階 1 で止める |

### 5.4 手順 1: サーバへの組み込み (段階 1)

1. `ArrayDataset::extents` と `Extents` を足し、npy に実装する。単体テストで
   「`extents` に従って読んだバイト列 == `read_range` の結果」を連続・非連続の選択で
   確かめる (proptest で選択を生成するとよい)
2. `reader.rs` に ring 版の inline モードを足す。`#[cfg(target_os = "linux")]`。既存の
   テスト (範囲を過不足なく覆う、失敗で止まる、失敗してもバッファが減らない、連続して
   要求を捌ける、drop で止まる) を ring 版でも通す。加えて、short read と、失敗時に
   投入済みの READ を刈り取ってからバッファを返すことのテスト
3. `io_engine` と `read_depth` を設定に足し、起動時の検出とログを入れる
4. 統合テスト (`tests/rust/`、`tests/python/`) を `AEX_IO_ENGINE=uring-read` と
   `legacy` の両方で通す。CI では両方を回す
5. mdx2 で `aexbench` を使い、[benchmark-mdx2.md](benchmark-mdx2.md) と同じ条件
   (4 GiB、チャンク 16 MiB) で `legacy` と比べる

### 5.5 手順 2: 段階 2

1. 接続の処理を状態機械として切り出す (段階 1 の成果を流用)
2. worker、ring 上の RECV / SEND、プールと枠、後始末を実装する。テストで確かめること:
   順序の約束 (FETCH を複数先行投入したときに DATA が到着順・offset 順で出る)、
   接続を途中で切ったときにバッファが漏れない・解放通知前に再利用されない、
   停止が全 worker で完了する、アイドルタイムアウトが効く
3. 統合テストを `uring-zc` / `uring-copy` / `legacy` で通す
4. mdx2 で `aexbench` (FETCH をまたいだ先読みの効果、credit を変えて) と
   `latbench` (小口読みの遅延が悪化していないこと) を測る
5. 多数接続の確認: 4 クライアント × 16 接続で、`legacy` と `uring-zc` のスループットと
   サーバのスレッド数・起床回数を比べる

### 5.6 結果の記録

`docs/benchmark-uring.md` に、既存の測定文書と同じ体裁 (結論を先に、測定環境、表、
再現方法) で書く。README の「状態」と「チューニング」節、SPEC の該当節 (§6.5 周辺) は、
既定値を変えるときにだけ更新する。

## 6. 作業上の約束 (Claude Code 向け)

- `CLAUDE.md` に従う。コメントは英語で短く、理由を書く。SPEC の節番号をコメントに
  書かない。1 コミット 1 機能で、各コミットがビルドとテストを通すこと
- チェック: `cargo fmt --all`、`cargo clippy --all-targets --all-features -- -D warnings`、
  `cargo test --all-features && cargo test --release --all-features`、
  `cargo check --all-targets`。**io_uring のコードは Linux でしかビルドされないので、
  macOS で開発している場合は Linux 機でもチェックを回すこと**
- io_uring 関連は `#[cfg(target_os = "linux")]` で囲み、macOS のビルドとテストを壊さない
- プロトコル (`aex-wire`) とクライアント (`aex-client`) は変更しない。変える必要が
  出たら、作業を止めて相談する
- 既定値 (`io_engine = auto` で何が選ばれるか、`read_depth`、`data_workers`) を変える
  変更は、測定結果を添えて提案し、独断で入れない
- unsafe は SQE の投入 (`push`) とバッファの生ポインタに限る。バッファが SQE から
  参照されている間は解放・再利用できないことを、型か少なくともアサーションで守る

## 7. 参考: 採らなかった選択肢

- **専用 I/O スレッド + channel、共有読みプール**: 受け渡しの費用が測定で確定している
- **tokio-uring / monoio / glommio**: データプレーンは async ランタイムを使わない設計。
  別ランタイムはコントロールプレーンの tokio とも噛み合わない
- **libaio**: バッファ I/O では `io_submit` の中で同期に読まれる。ソケットも扱えない。
  io_uring が使えない環境のフォールバックは `legacy` で足りる
- **glibc の POSIX AIO**: ユーザ空間のスレッドプールで pread を呼ぶだけで、pair と同じ
- **`sendfile`**: [sendfile.md](sendfile.md) の判断を維持。無加工の連続範囲では CPU 最小
  だが、適用範囲が狭く、コールドで重ならない

## 8. 未決事項とリスク

- **クライアントの順序の前提。** クライアントが FETCH をまたいだ DATA の入れ替えを
  受け付けないことを、`pool.rs` の in-flight の扱いで確認する (受け付けるなら、
  サーバの送信順の制約を緩められる)
- **ネットワーク FS での io-wq。** 非同期バッファ読みに対応しないファイルシステム
  (Lustre など) では READ が io-wq に回され、スレッド間のコピーに戻る。io-wq の CPU 時間を
  測り、必要なら `IORING_REGISTER_IOWQ_AFF` で worker と同じコア群に寄せる
- **受信側が次の律速になる。** 送信側を軽くしても接続 1 本の速度が伸びない可能性が高い。
  その場合の価値は「同じコア数でより多くの接続を捌けること」に移る
- **memlock。** 登録バッファとゼロコピー送信のページ固定は memlock に計上される。
  運用手順 (`LimitMEMLOCK=` など) を README に書く必要がある
- **O_DIRECT。** 大きな連続転送では、コールドのページキャッシュを経由するコピーと
  キャッシュ汚染を避けられる。一方で対話的な再読みでキャッシュが効かなくなる。
  `extents` とページ境界に揃えたバッファ配置で後から入れられるようにしておき、
  採否はこの計画の測定後に決める
- **5.15 系の不具合。** OCTOPUS のカーネルが 5.15.x のどの版かを記録する。
  io_uring の修正は安定版に多数入っている
