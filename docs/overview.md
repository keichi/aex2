# 実装の概要

README は利用者向けの手順だけを載せている。ここには実装の状態・クレートの役割・
現時点での制限・前提・測定結果の一覧を置く。設計の全体像とプロトコル仕様は
[SPEC.md](../SPEC.md) を参照のこと。

## 状態

**M6 (評価) まで実装済み。** Python と Rust の両方から
`.npy`、HDF5 (netCDF-4 を含む)、Zarr v3 (sharding を含む) の任意の選択を取得できる。実データは protobuf を
一切通らず、カーネルから呼び出し側のバッファ (Python では `np.empty` した配列) へ
直接読み込まれる。1 クライアントが複数のデータ接続を使い、VM 間の実ネットワークでは
16 接続で 1 接続の 8 倍 (163 Gbit/s) 出る ([M4 の測定](benchmark-m4.md))。

**既定値は実測で決めてある** ([M6 の測定](benchmark-m6.md))。VM 間の 1 GiB を
Python から読むと、前身の v1 が 456 MiB/s のところ 10,647 MiB/s (23.3 倍) 出る。

小さい選択は 1 往復で返り (`inline_limit_bytes` 以下)、`gather` は N 個の選択を
1 往復にまとめる。往復 100 ms で 64 個なら 6,460 ms が 102 ms になる
([gather の測定](benchmark-gather.md))。`np.sum` などの集約はサーバで計算する。

## クレートの役割

- `aex-core` — `DType`、`ErrorClass` / `AexError`、選択の解決 (`Index` の正規化と
  `SelectionLayout`)、numpy と同じ意味論の集約 (`reduce`)、`ArrayFile` /
  `ArrayDataset` トレイト、`.npy` / Zarr v3 バックエンド、HDF5 バックエンド
  (`hdf5` feature)、チャンク格子の走査 (`backends/chunks.rs`、HDF5 と Zarr で共用)
- `aex-wire` — データプレーンのワイヤ形式。ハンドシェイク、32 バイト固定ヘッダの
  フレーム、複数接続が 1 個の出力バッファへ書き込むための `ScatterBuffer`。
  サーバとクライアントが**同一コードから**エンコード/デコードする
- `aex-proto` — `protos/aex.proto` から tonic/prost が生成するコードと、
  生成型とコア型の相互変換
- `aex-server` — コントロールプレーン (セッション、ファイル、メタデータ、
  `PrepareSelection` / `PrepareSelections`、`ApplyFunction` による集約)、
  `TransferRegistry`、接続ごとに読みスレッドと送りスレッドを持つデータプレーン
- `aex-client` — 同期 API の Rust クライアント。`prepare` / `fill`、
  `prepare_many` / `fill_many`、`apply_function`、`stats`、
  `read_selection_into` / `read_selection` / `read_selection_as`。転送は
  `granted_streams` 本の接続へワークスティーリングで配り、各接続は credit 個まで
  `FETCH` を先行投入する。切れた接続のチャンクだけを別接続で再取得する
- `aex-py` — PyO3 による拡張モジュール `aex._aex`。ネットワーク待ちの間は GIL を解放し、
  出力配列を検証してから直接書き込む
- `python/aex` — ほぼ v1 と同じ API (`Client` / `FileProxy` / `GroupProxy` / `ArrayProxy`)。
  v1 にない `arr[..., 0]`、`arr[:, None]`、boolean mask にも対応。`GroupProxy` は
  h5py と同じ `Mapping` で、反復は子の名前を返す (v1 はプロキシを返した)。性能用 API として
  `read_into` / `gather` / `get_async` / `at()` (適応品質の枠) / `client.stats()`。
  `np.sum` などの集約 (SPEC §5.8) はサーバで計算し、`arr.view[0:100]` で転送せずに
  選択へ集約できる

## この時点での制限

今後解消する予定の、**実装上の**制限 (プロトコルの制限ではない)。

- **非連続な選択は連続読みより遅い**。近接した断片はまとめて `pread` する
  (隙間 4 KiB 以下、1 回 1 MiB まで) が、切り出しは要素単位で歩くため、
  `arr[::2]` (float32) は 1 接続で約 600 MiB/s に留まる
- **集約以外の numpy 関数はクライアントで計算する**。配列全体を転送してから計算する
  (64 MiB を超えると `AexFallbackWarning`)。集約でも結果が 64 KiB を超えるもの、
  `dtype` / `out` / `where` / `initial` を指定したものは同様にローカルで計算する
- **サーバの集約は 1 スレッドで逐次読む**。float の総和は numpy (pairwise) と
  ビット単位では一致しない
- **適応品質は誤差上限のみ**。`sz` / `zfp` feature を有効にしたビルドで
  `at(abs_error=...)` が誤差保証圧縮になる。コーデックは転送ごとに選べ
  (`at(abs_error=1e-3, codec="zfp")`)、実際に使われたものが
  `applied_quality["codec"]` に返る。既定は SZ3
- **無損失より速く届くのは、帯域が 4.7 〜 31.9 Gbit/s より狭い回線**
  (誤差上限・接続数・コーデックによる)。1 Gbit/s なら SZ3 で 4.6 〜 13.8 倍
  ([測定](benchmark-sz.md))
- **2 つのコーデックは狭い回線と広い回線で勝ち負けが入れ替わる**。圧縮率は
  SZ3 が 4.83 〜 72.7 倍、ZFP は 1.64 〜 5.82 倍。速度は逆に ZFP が 1.2 〜 2.4 倍。
  狭い回線では圧縮率が、広い回線では CPU が単独で効くので、2.5 〜 7.4 Gbit/s で
  入れ替わる: 1 Gbit/s では SZ3 が 2.8 倍速く、26 Gbit/s では ZFP が 2.3 倍速い
  ([比較](benchmark-zfp.md))。ZFP は誤差上限を 2 の冪に切り下げて守り、
  **NaN / Inf を含む配列では上限を保証しない** (SZ3 にこの制限は無い)
- **どちらのコーデックも単スレッドで走る**ので、圧縮転送では `streams` がそのまま
  何コアで圧縮するかを決める。既定の 8 は無損失転送に合わせた値で、コア数まで
  上げると 1.7 〜 1.9 倍になる。dtype キャスト・間引き・値域相対の誤差、および
  float32 / float64 以外の dtype は EXACT で返し、`AexQualityWarning` を出す。
  既定ビルドと Python の wheel にはどちらのコーデックも入っていない
- **表現できない型の属性は `.attrs` に現れない**。数値と文字列は載るが、compound、
  enum、オブジェクト参照、opaque、文字列の配列、および 64 KiB を超えるものは
  黙って落ちる。netCDF-4 が書く `CLASS` / `NAME` / `_Netcdf4Dimid` などの内部属性は
  h5py と同じく見えたままで、隠すのは上の層の仕事とした
- **変数の次元名は出していない**。`.attrs` に `_Netcdf4Coordinates` (次元 id) は
  届くので、次元データセットの `_Netcdf4Dimid` と突き合わせれば名前は組めるが、
  その対応づけはまだ実装していない
- **多次元の整数インデックス配列は非対応**。1 次元にして送り、結果を reshape すること
- **credit は固定値** (既定 16、`AEX_CREDIT`)。RTT と帯域から自動で決める処理は
  入れていない。**遅延のある回線では `streams × credit × chunk_bytes` が転送の
  頭打ちを決める** ([遅延を足した測定](benchmark-delay.md)、
  [M6 の測定](benchmark-m6.md))
- **`get_async` は転送を中断できない**。`Future` を捨てても転送は最後まで走る
- `tcp.congestion` は Linux でのみ適用する (他の OS では起動時に警告を出す)

## 前提と制約

- 対象 OS は Linux (最適化対象) と macOS。`pread` を使うため Unix 系に限る
- 対応形式は `.npy`、HDF5 (netCDF-4 を含む)、Zarr v3。Zarr v2 と netCDF-3 は
  将来課題 (SPEC §14.1)
- Zarr のコーデックは `bytes` / 恒等 `transpose` / gzip / zstd / crc32c と
  `sharding_indexed`。blosc と入れ子の shard は未対応で、open 時に理由を付けて
  拒否する。ストアはローカルディレクトリのみで、S3/HTTP ストアは扱わない
- Zarr の属性は JSON なので要素型が無い。整数は `INT64`、他の数値は `FLOAT64` に
  なる。文字列の配列や入れ子のオブジェクトは HDF5 と同じく黙って落ちる
- HDF5 の圧縮フィルタは deflate / shuffle / fletcher32 のみ。compact・virtual・
  外部ファイル格納のデータセットと external link は非対応 (SPEC §7.5)
- 読み出し専用
- fortran order の `.npy` と、ビッグエンディアンのデータは非対応 (SPEC §7.2, §7.5)
- データプレーンは平文。認証はセッショントークンと転送ごとの ticket のみで、
  「少数クライアント・信頼できる環境」を前提とする。厳密なマルチテナント制御や
  暗号化は行わない (TLS は HELLO の `flags` に枠のみ確保)
- 接続ごとに専用 OS スレッドを使う。スレッド数は `クライアント数 × 接続数` に
  比例する

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

サーバは接続ごとに 1 スレッドで読んで送り、次の piece をカーネルに先読みさせて
ディスクとネットワークを重ねる (Linux のみ。他の OS では何もしない)。コールドな
ディスクでは 1 接続で 1,081 → 1,296 MiB/s、4 接続で 1,699 → 2,143 MiB/s になる
([先読みの測定](benchmark-fadvise.md))。読みスレッドを別に立てる方式は、
同じスレッド数を接続数に使ったほうが速いので採らない。

## リポジトリ構成

```
protos/aex.proto   コントロールプレーンの定義
crates/aex-core/   共通型・選択の解決・バックエンド (tokio / tonic に依存しない)
                   `.npy`、HDF5、Zarr、転送経路の測定用の合成バックエンド
crates/aex-wire/   データプレーンのワイヤ形式 (サーバとクライアントで共用)
crates/aex-proto/  aex.proto から生成されるコードと型変換
crates/aex-server/ サーバ (両プレーン)
crates/aex-client/ Rust クライアント
crates/aex-py/     Python 拡張モジュール aex._aex
python/aex/        Python 層 (v1 互換の API)
benchmarks/        転送性能の測定 (基準値の iperf3 / pread、端から端まで、v1 との比較)
docs/              測定結果
tests/rust/        サーバとクライアントを同一プロセスで動かす統合テスト
tests/python/      v1 から移植した pytest と numpy との差分テスト
```

## 測定結果

| 文書 | 環境 | 分かったこと |
|---|---|---|
| [パラメータ掃引と既定値](benchmark-m6.md) | VM 間 | 既定値を streams 8・credit 16・read_buffers 1 に決めた。1 GiB が v1 の 23.3 倍 |
| [並列ストリーム](benchmark-m4.md) | VM 間・ローカル | 接続数にほぼ線形。16 本で 1 本の 8.0 倍 (163 Gbit/s) |
| [VM 間 (実ネットワーク)](benchmark-mdx2.md) | VM 間 | 実ネットワークでの単一接続。制御プレーンの Nagle を切って小口読みが 383 倍 |
| [遅延を足した場合](benchmark-delay.md) | VM 間 + netem | 遅延のある回線の上限は in-flight バイト数で決まる |
| [gather が省く往復](benchmark-gather.md) | VM 間 + netem | 往復 100 ms で 64 個なら 6,460 ms が 102 ms (63.5 倍) |
| [HDF5 バックエンド](benchmark-hdf5.md) | VM 間 | contiguous な HDF5 は `.npy` と同速。gzip は伸長が律速 |
| [Zarr と sharding](benchmark-zarr.md) | VM 間 | チャンクを fetch に合わせるかどうかで 5.4 倍。差は 1 要求が使えるコア数 (14.4 対 3)。チャンクごとの 4 MiB 確保し直しが 23 % を食っている |
| [誤差保証圧縮 (SZ3)](benchmark-sz.md) | VM 間 + tc | 勝ち負けを決めるのは帯域。1 Gbit/s で 4.6 〜 13.8 倍 |
| [SZ3 と ZFP の比較](benchmark-zfp.md) | VM 間 + tc | 圧縮率は SZ3、速度は ZFP。2.5 〜 7.4 Gbit/s で入れ替わる |
| [Linux での転送性能](benchmark-linux.md) | Linux 機 | Linux ではソケットが律速。実ネットワークに近いのはこちら |
| [v1 との比較](benchmark-m3-v1-v2.md) | Mac・VM | Python から測った v1 との比較。以降の最適化はここを起点に追う |
| [M2 時点の転送性能](benchmark-m2-local.md) | ローカル | 単一接続の律速をメモリ帯域と 1 チャンクあたりの往復に分解した |
| [ダブルバッファリング](benchmark-double-buffering.md) | ローカル | 読みと送りを分けてメモリ上のデータが 7,966 → 12,048 MiB/s |
| [ストレージを外した場合](benchmark-null-backend.md) | ローカル | 合成バックエンドで転送経路だけを測り、ストレージの取り分を出した |
| [先読みの効果](benchmark-fadvise.md) | ローカル | コールドディスクで 1 接続 1,081 → 1,296 MiB/s。読みスレッドは不要 |
| [共有読みプールの予備測定](benchmark-read-pool.md) | ローカル | 読みプールより、同じスレッド数を接続に使うほうが速い |
| [io_uring を採らない理由](benchmark-uring.md) | ローカル | 得られるものは `posix_fadvise` 10 行で同じ。メモリ常駐では負ける |
| [sendfile を採らない理由](sendfile.md) | ローカル | macOS では効かず、Linux でもメモリ常駐に限る |
| [非連続選択の歩き方](benchmark-gather-walk.md) | マイクロベンチ | 内側次元を 1 本のループで歩いて 2.7 〜 6.9 倍 |

mdx2 での測り方の決まりごとは [eval-mdx2.md](eval-mdx2.md) にまとめた。

macOS (M4) ではメモリ上のデータで単一接続 12,048 MiB/s (iPerf3 の 62 %)、Linux
(Ryzen 9 5900X) では 6,694 MiB/s (iPerf3 単一ストリームは 5,863 MiB/s)。どちらも
単一接続で、そこではダブルバッファリングが 50 % 以上効く (接続を増やすと逆転する。
[M6 の測定](benchmark-m6.md))。
**Linux では読みスレッドと送りスレッドを同じ L3 に載せるかどうかで 36 % 変わる**
(未対応)。
