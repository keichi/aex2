# AEX2

**Array Exchange v2** — 広域ネットワーク上に分散した計算資源の間で、配列データを
部分的・オンデマンドに転送するためのデータ流通基盤。

Rust 実装で、メタデータ操作を担う**コントロールプレーン** (gRPC) と、実データを運ぶ
**データプレーン** (生 TCP の独自フレームプロトコル) を分離する。前身は
[`aex`](https://github.com/keichi/aex) (Python + gRPC)。

設計の全体像・プロトコル仕様・マイルストーンは [SPEC.md](SPEC.md) を参照のこと。

## 状態

**M6 (評価) まで実装済み。** Python と Rust の両方から
`.npy` と HDF5 (netCDF-4 を含む) の任意の選択を取得できる。実データは protobuf を
一切通らず、カーネルから呼び出し側のバッファ (Python では `np.empty` した配列) へ
直接読み込まれる。1 クライアントが複数のデータ接続を使い、VM 間の実ネットワークでは
16 接続で 1 接続の 8 倍 (163 Gbit/s) 出る ([M4 の測定](docs/benchmark-m4.md))。

**既定値は実測で決めてある** ([M6 の測定](docs/benchmark-m6.md))。VM 間の 1 GiB を
Python から読むと、前身の v1 が 456 MiB/s のところ 10,647 MiB/s (23.3 倍) 出る。

小さい選択は 1 往復で返り (`inline_limit_bytes` 以下)、`gather` は N 個の選択を
1 往復にまとめる。往復 100 ms で 64 個なら 6,460 ms が 102 ms になる
([gather の測定](docs/benchmark-gather.md))。`np.sum` などの集約はサーバで計算する。

- `aex-core` — `DType`、`ErrorClass` / `AexError`、選択の解決 (`Index` の正規化と
  `SelectionLayout`)、numpy と同じ意味論の集約 (`reduce`)、`ArrayFile` /
  `ArrayDataset` トレイト、`.npy` バックエンド、HDF5 バックエンド (`hdf5` feature)
- `aex-wire` — データプレーンのワイヤ形式。ハンドシェイク、32 バイト固定ヘッダの
  フレーム、複数接続が 1 個の出力バッファへ書き込むための `ScatterBuffer`。
  サーバとクライアントが**同一コードから**エンコード/デコードする
- `aex-proto` — `protos/aex.proto` から tonic/prost が生成するコードと、
  生成型とコア型の相互変換
- `aex-server` — コントロールプレーン (セッション、ファイル、メタデータ、
  `PrepareSelection` / `PrepareSelections`、`ApplyFunction` による集約)、`TransferRegistry`、接続ごとに読みスレッドと送りスレッドを
  持つデータプレーン
- `aex-client` — 同期 API の Rust クライアント。`prepare` / `fill`、
  `prepare_many` / `fill_many`、`apply_function`、`stats`、
  `read_selection_into` / `read_selection` / `read_selection_as`。転送は
  `granted_streams` 本の接続へワークスティーリングで配り、各接続は credit 個まで
  `FETCH` を先行投入する。切れた接続のチャンクだけを別接続で再取得する
- `aex-py` — PyO3 による拡張モジュール `aex._aex`。ネットワーク待ちの間は GIL を解放し、
  出力配列を検証してから直接書き込む
- `python/aex` — v1 と同じ API (`Client` / `FileProxy` / `GroupProxy` / `ArrayProxy`)。
  v1 にない `arr[..., 0]`、`arr[:, None]`、boolean mask にも対応。性能用 API として
  `read_into` / `gather` / `get_async` / `at()` (適応品質の枠) / `client.stats()`。
  `np.sum` などの集約 (SPEC §5.8) はサーバで計算し、`arr.view[0:100]` で転送せずに
  選択へ集約できる

### この時点での制限

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
  ([測定](docs/benchmark-sz.md))
- **2 つのコーデックは狭い回線と広い回線で勝ち負けが入れ替わる**。圧縮率は
  SZ3 が 4.83 〜 72.7 倍、ZFP は 1.64 〜 5.82 倍。速度は逆に ZFP が 1.2 〜 2.4 倍。
  狭い回線では圧縮率が、広い回線では CPU が単独で効くので、2.5 〜 7.4 Gbit/s で
  入れ替わる: 1 Gbit/s では SZ3 が 2.8 倍速く、26 Gbit/s では ZFP が 2.3 倍速い
  ([比較](docs/benchmark-zfp.md))。ZFP は誤差上限を 2 の冪に切り下げて守り、
  **NaN / Inf を含む配列では上限を保証しない** (SZ3 にこの制限は無い)
- **どちらのコーデックも単スレッドで走る**ので、圧縮転送では `streams` がそのまま
  何コアで圧縮するかを決める。既定の 8 は無損失転送に合わせた値で、コア数まで
  上げると 1.7 〜 1.9 倍になる。dtype キャスト・間引き・値域相対の誤差、および
  float32 / float64 以外の dtype は EXACT で返し、`AexQualityWarning` を出す。
  既定ビルドと Python の wheel にはどちらのコーデックも入っていない
- **多次元の整数インデックス配列は非対応**。1 次元にして送り、結果を reshape すること
- **credit は固定値** (既定 16、`AEX_CREDIT`)。RTT と帯域から自動で決める処理は
  入れていない。**遅延のある回線では `streams × credit × chunk_bytes` が転送の
  頭打ちを決める** ([遅延を足した測定](docs/benchmark-delay.md)、
  [M6 の測定](docs/benchmark-m6.md))
- **`get_async` は転送を中断できない**。`Future` を捨てても転送は最後まで走る
- `tcp.congestion` は Linux でのみ適用する (他の OS では起動時に警告を出す)

ローカル (同一ホスト) での転送性能の測定結果は `docs/` にある
([M6: パラメータ掃引と既定値](docs/benchmark-m6.md)、[M4: 並列ストリーム](docs/benchmark-m4.md)、[共有読みプールの予備測定](docs/benchmark-read-pool.md)、[v1 との比較 (Mac・VM)](docs/benchmark-m3-v1-v2.md)、[M2 時点](docs/benchmark-m2-local.md)、[ダブルバッファリング](docs/benchmark-double-buffering.md)、
[ストレージを外した場合](docs/benchmark-null-backend.md)、
[sendfile を採らない理由](docs/sendfile.md)、
[io_uring を採らない理由](docs/benchmark-uring.md)、
[先読みの効果](docs/benchmark-fadvise.md)、
[非連続選択の歩き方](docs/benchmark-gather-walk.md)、
[誤差保証圧縮 (SZ3)](docs/benchmark-sz.md)、
[SZ3 と ZFP の比較](docs/benchmark-zfp.md))。Linux 機での測定は
[docs/benchmark-linux.md](docs/benchmark-linux.md)、mdx2 の VM 2 台を実ネットワークで
繋いだ測定は [docs/benchmark-mdx2.md](docs/benchmark-mdx2.md)、そこに遅延を足した測定は
[docs/benchmark-delay.md](docs/benchmark-delay.md)、gather が省く往復の測定は
[docs/benchmark-gather.md](docs/benchmark-gather.md)、HDF5 バックエンドの測定は
[docs/benchmark-hdf5.md](docs/benchmark-hdf5.md) にある。mdx2 での測り方の決まりごとは
[docs/eval-mdx2.md](docs/eval-mdx2.md) にまとめた。

macOS (M4) ではメモリ上のデータで単一接続 12,048 MiB/s (iPerf3 の 62 %)、Linux
(Ryzen 9 5900X) では 6,694 MiB/s (iPerf3 単一ストリームは 5,863 MiB/s)。どちらも
単一接続で、そこではダブルバッファリングが 50 % 以上効く (接続を増やすと逆転する。
[M6 の測定](docs/benchmark-m6.md))。
**Linux では読みスレッドと送りスレッドを同じ L3 に載せるかどうかで 36 % 変わる**
(未対応)。

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

## ビルドとテスト

`protoc` が必要である (`brew install protobuf` / `apt install protobuf-compiler`)。
HDF5 バックエンド (`hdf5` feature) には libhdf5 1.14 以降が要る
(`brew install hdf5` / `apt install libhdf5-dev`。Ubuntu は 26.04 以降)。

誤差保証圧縮 (`sz` / `zfp` feature) は SZ3 と zfp をソースからビルドするため、
cmake・C++17 コンパイラ・libclang が要る (`brew install cmake` /
`apt install cmake libclang-dev`)。どちらも既定では無効で、そのぶん素のビルドは
これらを必要としない。リンクする `sz3-sys` は GPL-3.0-only である (SZ3 本体は
BSD)。`zfp-sys` は MIT で、静的リンクしている。Python から使うには拡張モジュールを
`maturin develop --features aex-py/sz,aex-py/zfp` でビルドする。

```console
$ cargo test --all-features
$ cargo clippy --all-targets --all-features -- -D warnings
$ cargo fmt --all -- --check
```

デバッグとリリースで挙動が変わる箇所 (オーバーフロー検査) があるため、CI は
`cargo test` を両プロファイルで実行する。

Python 側は拡張モジュールをビルドしてから pytest を実行する。テストは
`aex-server` を `hdf5` feature 付きで自分でビルドして起動する。

```console
$ pip install -e ".[dev]"        # maturin で aex._aex をビルド
$ pytest
$ mypy && ruff check python tests/python benchmarks
```

`protoc` は **3.15 以降**が要る (proto3 の optional フィールドを使うため)。
Ubuntu 22.04 の `protobuf-compiler` は 3.12 なので、
[公式リリース](https://github.com/protocolbuffers/protobuf/releases) から入れること。

## 使い方

```console
$ cargo install --path crates/aex-server --features hdf5   # HDF5 が不要なら --features なし
$ aex-server --control-addr 127.0.0.1:50051 --data-addr 127.0.0.1:50052 \
    --root /path/to/data
```

形式はファイルの拡張子で決まる (`.npy`、`.h5` / `.hdf5` / `.he5` / `.nc`)。
拡張子で判別できないファイルは、Rust クライアントの `open_as` で形式名
(`npy`、`hdf5`、`netcdf4` など) を明示して開く。

設定ファイルを渡すこともできる (指定しなかった項目は既定値のまま)。項目は
SPEC §8.2 を参照のこと。

```console
$ aex-server --config server.toml
```

```rust
use aex_client::{Client, ClientConfig, FunctionArg, Index, Item, Selection};

let client = Client::connect("http://127.0.0.1:50051", ClientConfig::default())?;
let handle = client.open("ocean.npy")?;           // データルート配下の相対パス

if let Item::Dataset(info) = client.get_item(handle, "array")? {
    println!("{:?} {:?}", info.dtype, info.shape);
}

// 配列全体、または先頭軸の連続した範囲
let all = client.read_selection_as::<f32>(handle, "array", &[])?;
let rows = client.read_selection_as::<f32>(handle, "array", &[Index::range(10, 20)])?;
println!("{:?} {:.1} MiB/s", rows.shape, rows.transfer.throughput_mib_per_sec());

// 呼び出し側のバッファへ直接読む (コピーが 1 回も入らない経路)
let mut buf = vec![0u8; all.data.len() * size_of::<f32>()];
client.read_selection_into(handle, "array", &[], &mut buf)?;

// 複数の選択を 1 往復で解決し、1 つのキューでまとめて受け取る
let keys = [vec![Index::range(0, 10)], vec![Index::range(90, 100)]];
let selections: Vec<_> = keys
    .iter()
    .map(|indices| Selection::exact(handle, "array", indices))
    .collect();
let plans: Vec<_> = client
    .prepare_many(&selections)?
    .into_iter()
    .collect::<Result<_, _>>()?;
let mut bufs: Vec<Vec<u8>> = plans.iter().map(|p| vec![0u8; p.total_bytes as usize]).collect();
let mut dsts: Vec<&mut [u8]> = bufs.iter_mut().map(|b| b.as_mut_slice()).collect();
client.fill_many(&plans, &selections, &mut dsts)?;

// 集約はサーバで計算する (結果だけが返る)
let total = client.apply_function(&selections[0], "sum", &[("axis", FunctionArg::None)])?;
println!("{:?} {:?}", total.dtype, total.shape);

client.disconnect()?;
```

Python からは v1 と同じ書き方で使える。

```python
import aex
import numpy as np

with aex.Client("127.0.0.1:50051") as client:
    arr = client.open("ocean.npy")["array"]   # ArrayProxy
    buf = np.empty((1000, 200), np.float32)
    arr[10:20, ::2]                           # 読み取り専用の ndarray
    arr[arr[:, 0] > 0]                        # boolean mask
    np.mean(arr, axis=0)                      # サーバで計算し、結果だけが返る
    np.max(arr.view[10:90], axis=1)           # 選択も転送せずに集約できる

    arr.read_into(buf, np.s_[0:1000])         # 確保済みのバッファへ
    a, b = arr.gather([np.s_[0:10], np.s_[90:100]])   # まとめて 1 往復
    future = arr.get_async(np.s_[:])          # 転送と計算を重ねる
    client.stats()                            # 転送量・往復時間などの累計
```

ベンチマークのパラメータ掃引のため、主要な設定は環境変数でも上書きできる
(`AEX_STREAMS`、`AEX_CHUNK_BYTES`、`AEX_MAX_RETRIES`、`AEX_TCP_NODELAY` ほか)。
ssh トンネルなどでサーバが広告するデータプレーンのポートに直接届かない場合は、
`AEX_DATA_ENDPOINT=host:port` で接続先を指定する。

## チューニング

既定値はそのままで使えるように決めてある ([M6 の測定](docs/benchmark-m6.md))。
それでも足りないときに動かす順序は次のとおり。

| | 既定 | 環境変数 |
|---|---|---|
| データ接続数 | 8 | `AEX_STREAMS` |
| credit (接続あたりの先行 `FETCH` 数) | 16 | `AEX_CREDIT` |
| チャンクサイズ | サーバ推奨値 (4 MiB) | `AEX_CHUNK_BYTES` |

1. **まず接続数を上げる。** どの往復時間でも効き、遅延のない回線では**唯一**効く。
   パイプラインに隠す往復がないので、そこでは credit もチャンクも効かない。
   上限はサーバの `limits.max_streams_per_session` (既定 32)
2. **接続数を増やせないぶんを credit で埋める。** 遅延のある回線でのスループットは
   接続あたりの in-flight バイト数 `streams × credit × chunk_bytes` だけで決まり、
   どのノブで作っても同じ値になる。帯域遅延積の 2〜4 倍を目安にする
   (20 Gbit/s × 50 ms に対し、in-flight 128 MiB で 1,753 MiB/s、256 MiB で 2,806)
3. **チャンクサイズは触らない。** credit と等価な上、大きくすると最初のフレームまでの
   待ちと再送の単位が増える
4. **`SO_RCVBUF` (`AEX_RCVBUF`) は設定しない。** 明示するとカーネルの自動調整が止まり、
   指定した値が窓の上限になる。往復 50 ms で 4 MiB を指定すると 1,028 MiB/s が
   327 MiB/s に落ちる。窓を広げたいなら `net.ipv4.tcp_rmem` の上限を上げる

サーバは接続ごとに 1 スレッドで読んで送り、次の piece をカーネルに先読みさせて
ディスクとネットワークを重ねる (Linux のみ。他の OS では何もしない)。コールドな
ディスクでは 1 接続で 1,081 → 1,296 MiB/s、4 接続で 1,699 → 2,143 MiB/s になる
([先読みの測定](docs/benchmark-fadvise.md))。読みスレッドを別に立てる方式は、
同じスレッド数を接続数に使ったほうが速いので採らない。

## 前提と制約

- 対象 OS は Linux (最適化対象) と macOS。`pread` を使うため Unix 系に限る
- 対応形式は `.npy` と HDF5 (netCDF-4 を含む)。netCDF-3 と Zarr は将来課題 (SPEC §14.1)
- HDF5 の圧縮フィルタは deflate / shuffle / fletcher32 のみ。compact・virtual・
  外部ファイル格納のデータセットと external link は非対応 (SPEC §7.5)
- 読み出し専用
- fortran order の `.npy` と、ビッグエンディアンのデータは非対応 (SPEC §7.2, §7.5)
- データプレーンは平文。認証はセッショントークンと転送ごとの ticket のみで、
  「少数クライアント・信頼できる環境」を前提とする。厳密なマルチテナント制御や
  暗号化は行わない (TLS は HELLO の `flags` に枠のみ確保)
- 接続ごとに専用 OS スレッドを使う。スレッド数は `クライアント数 × 接続数` に
  比例する

## リポジトリ構成

```
protos/aex.proto   コントロールプレーンの定義
crates/aex-core/   共通型・選択の解決・バックエンド (tokio / tonic に依存しない)
                   `.npy`、HDF5、転送経路の測定用の合成バックエンド
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
