# AEX2 設計書

**Array Exchange v2 — Rust 実装・コントロール/データプレーン分離版**

- 対象プロジェクト: 探索的データ分析のための広域科学データ流通基盤 (JST ACT-X / AI共生社会を拓くサイバーインフラストラクチャ)
- 前身: `/Users/keichi/Projects/aex` (Python + gRPC 実装)
- 本リポジトリ: `/Users/keichi/Projects/aex2` (新規)
- 状態: 設計フェーズ / 未実装

---

## 1. 背景と目的

### 1.1 研究上の位置づけ

本基盤は「広域ネットワーク上に分散した計算資源の間で、**必要なデータ**を**必要なとき**に**必要な品質**で移動させる」ことを目的とする。ファイル単位でしかデータを扱えない既存のデータ流通基盤に対し、配列データの部分アクセス・オンデマンド転送・適応品質転送を提供する。

AEX2 はその実装基盤であり、本設計書は「細粒度アクセスの性能をネットワーク帯域の限界まで引き上げる」ことを主目的とする。

### 1.2 v1 (Python 実装) の到達点と限界

v1 は機能面では以下を達成している。

- gRPC による配列の部分取得 (`GetSelection`)、階層探索、サーバサイド numpy 関数実行 (`ApplyFunction`)
- 4 バックエンド (NumPy `.npy` / HDF5 / netCDF4 / Zarr) のプラグイン抽象化
- `__getitem__` / `__array_function__` による numpy 互換のクライアント API

一方、転送性能が要求水準に達していない。`docs/throughput-research.md` の調査および現行コードの読解から、律速要因と AEX2 での対処は以下と整理される。

| # | v1 の律速要因 | 影響 | v2 の対処 |
|---|--------------|------|-----------|
| 1 | `array.tobytes()` の全データコピー (`server.py:_stream_array`) | 1 GB 転送で 1 GB の追加確保と memcpy | `pread` で読みバッファへ直接読み、バッファを使い回す |
| 2 | 1 MiB 固定チャンクの Python ループ (同上) | 1 GB で 1024 回の protobuf メッセージ構築 | 4 MiB 既定 + Rust。チャンク数 1/4、ループコストを 2 桁削減 |
| 3 | 受信側で `bytearray` へ逐次コピー後に `np.frombuffer` (`array_proxy.py`) | 受信側でもう 1 回の全コピー | `ScatterBuffer` で numpy 配列へ直接 `read_exact` |
| 4 | GIL | 並列ストリーム化してもスケールしない | Rust + `allow_threads`。接続ごとに独立した OS スレッド |
| 5 | HTTP/2 フロー制御ウィンドウ未調整 | 初期ウィンドウ 64 KB。高 BDP 環境で帯域の数 % しか使えない | 生 TCP。フロー制御は TCP のみ。ウィンドウはカーネルの自動調整に任せる |
| 6 | 1 リクエスト = 1 RTT の完全同期 (`__getitem__`) | 対話的分析で RTT が積算、WAN で致命的 | `gather` / `get_async` / credit ベースのパイプライン |
| 7 | 単一 TCP コネクション | 単一フローでは高 BDP 環境でリンクを飽和できない | 常設プール N 本の並列ストリーム |
| 8 | protobuf の `bytes` フィールド経由 | ペイロードがフレーム境界に整列せず、ゼロコピー受信ができない | 32 バイト固定ヘッダ + 生ペイロード |

1〜5 は v1 のままでも緩和できるが、**6〜8 はプロトコル設計に起因するため、実装言語を変えるだけでは解決しない**。AEX2 はここを設計からやり直す。

### 1.3 データパスの比較

1 GB の連続選択を取得する場合の、全データ分のコピー回数。

**v1**

```
HDF5/npy (mmap)
  → numpy スライス       [コピー 1: page cache → 新規配列]
  → tobytes()            [コピー 2: ユーザ空間内]
  → protobuf メッセージ  [コピー 3: ユーザ空間内、1024 メッセージに分割]
  → gRPC/HTTP2 → TCP     [コピー 4: ユーザ空間 → カーネル]
  ─────────── network ───────────
  → gRPC 受信バッファ    [コピー 5: カーネル → ユーザ空間]
  → bytearray            [コピー 6: ユーザ空間内]
  → np.frombuffer        [コピーなし: ビュー]
合計 6 回 (うちユーザ空間内が 3 回) + Python の 1024 回ループ。1 GB の一時バッファを 3 枚確保。
```

**v2**

```
npy (File)
  → pread(fd, &mut buf, src_offset)   [コピー 1: カーネル → 読みバッファ]
  → writev(header, &buf[..len])       [コピー 2: 読みバッファ → カーネル]
  ─────────── network ───────────
  → read_exact(&mut np_array[off..])  [コピー 3: カーネル → 出力 numpy 配列]
合計 3 回。**ユーザ空間内のコピーはゼロ** (すべてカーネル境界をまたぐもので、
TCP を使う限り最低 2 回は必須)。読みバッファは使い回すためアロケーションもゼロ。
加えて並列 4 ストリーム。
```

なお、Python API を v1 と互換に保つことで、同一のベンチマークスクリプトから両者を測定し、
改善幅を定量比較できるようにする (第 12 章)。

---

## 2. 設計上の決定事項

本設計書を書くにあたって確定した方針を先に列挙する。

| 項目 | 決定 | 備考 |
|------|------|------|
| リポジトリ | `aex2` を新規に立ち上げ | `aex` は比較用ベースラインとして保存 |
| サーバ/クライアント実装言語 | Rust | |
| Python 連携 | PyO3 + maturin | `rust-numpy` で ndarray 相互運用 |
| コントロールプレーン | gRPC (tonic) を維持 | v1 の `aex.proto` を拡張 |
| データプレーン | 生 TCP の独自フレームプロトコル | |
| データプレーン接続モデル | 常設コネクションプール + request_id 多重化 | §6.4 |
| データプレーン認証 | チケット (ワンタイムトークン) のみ、平文転送 | §6.2。TLS は枠のみ確保 |
| 対応フォーマット | **`.npy` のみ** | §7.1。トレイト設計は 4 種を見据える |
| 書き込み | 非対応 (読み出し専用) | プロトコルは双方向定義可能な形に |
| サーバサイド計算 | 主要な集約関数のみ | §5.8 |
| 可逆圧縮 | `GZIP` のみ (既定 ON、ベースライン用)。LZ4 / ZSTD は実装しない | §5.5.2、§14.1 |
| 適応品質 | 誤差上限を SZ3 と ZFP で実装 (`sz` / `zfp` feature、既定 OFF)。dtype キャストは既定 ON。値域相対は枠のみ | §5.5.1、§5.5.3、§14.1 |
| 対象 OS | Linux を最適化対象、macOS でも動作 | OS 固有機能は feature フラグで分離 |
| Python API | v1 の API を維持し、性能用 API を**追加** | §10.2、§10.3 |
| numpy 未対応関数 | 警告を出してローカルフォールバック | §10.4 |
| 想定規模 | 少数クライアント・信頼できる環境 | 厳密なマルチテナント制御は行わない |
| 性能検証環境 | 手元の Mac (localhost)、学内 LAN クラスタ | §12.2。WAN は将来 |

---

## 3. 全体アーキテクチャ

```
┌──────────────────────── Client (Rust + PyO3) ────────────────────────┐
│                                                                       │
│   Python層 (python/aex/)                                              │
│     Client / FileProxy / GroupProxy / ArrayProxy                      │
│     __getitem__, __array_function__, read_into, gather, get_async     │
│                          │  PyO3 (GIL 解放)                           │
│   Rust層 (aex-client)    ▼                                            │
│     ┌──────────────┐          ┌──────────────────────────────┐        │
│     │ ControlConn  │          │      DataPool                │        │
│     │ (tonic/HTTP2)│          │  ┌────┐┌────┐┌────┐┌────┐    │        │
│     └──────┬───────┘          │  │ S0 ││ S1 ││ S2 ││ S3 │... │        │
│            │                  │  └──┬─┘└──┬─┘└──┬─┘└──┬─┘    │        │
│            │                  └─────┼─────┼─────┼─────┼──────┘        │
└────────────┼────────────────────────┼─────┼─────┼─────┼───────────────┘
             │ gRPC                   │ 生TCP (並列・多重化)
             │ ・OpenFile             │ ・FETCH{request_id, range}
             │ ・GetItem              │ ・DATA{request_id, offset, len}
             │ ・PrepareSelection ────┼──→ TransferPlan{ticket, chunks}
             │ ・ApplyFunction        │
             ▼                        ▼
┌──────────────────────── Server (Rust) ───────────────────────────────┐
│   ┌────────────────────┐      ┌─────────────────────────────────┐    │
│   │ ControlService     │      │  DataPlane                      │    │
│   │ (tonic / tokio)    │      │  接続ごとに専用 OS スレッド      │    │
│   │  SessionRegistry   │─────→│  TransferRegistry               │    │
│   │  FileRegistry      │ plan │  pread → buf → writev(hdr, buf) │    │
│   └────────┬───────────┘      └────────────┬────────────────────┘    │
│            │                               │                          │
│            ▼                               ▼                          │
│   ┌──────────────────────────────────────────────────┐               │
│   │  Backend (trait ArrayFile / ArrayDataset)         │               │
│   │    NpyFile: pread + ヘッダ解析                    │               │
│   │    Hdf5File: libhdf5 はメタデータのみ + pread     │               │
│   │    ZarrFile: zarr.json 解析 + チャンクごとの read  │               │
│   └──────────────────────────────────────────────────┘               │
└───────────────────────────────────────────────────────────────────────┘
```

### 3.1 プレーン分離の要点

**コントロールプレーン (gRPC)** は「何を送るか」を決める。セッション確立、ファイル操作、メタデータ取得、選択の解決、転送計画の発行。メッセージは小さく往復も少ないため、gRPC の利便性 (スキーマ、コード生成、エラーモデル、将来の多言語クライアント) をそのまま享受できる。

**データプレーン (生 TCP)** は「どう送るか」だけを担う。設計原則は第 6.1 節に示す。

分離の本質的な利点は、**データが protobuf のエンコード/デコードを一切通らない**ことにある。Arrow Flight が `data_body` フィールドに対して行っている最適化を、独自プロトコルとして素直に実装する形になる。

---

## 4. リポジトリ構成

```
aex2/
├── Cargo.toml                    # workspace root
├── pyproject.toml                # maturin ビルド設定
├── SPEC.md                       # 本書
├── README.md
├── protos/
│   └── aex.proto                 # コントロールプレーン定義
├── crates/
│   ├── aex-core/                 # 共通型・バックエンドトレイト
│   │   └── src/
│   │       ├── dtype.rs          # DType 定義と numpy descr 変換
│   │       ├── selection.rs      # Selection / SelectionLayout / Fragment
│   │       ├── backend.rs        # ArrayFile / ArrayDataset トレイト
│   │       ├── backends/npy.rs   # .npy バックエンド
│   │       ├── backends/hdf5.rs  # HDF5 バックエンド (hdf5 feature)
│   │       ├── reduce.rs         # サーバサイド集約
│   │       ├── quality.rs       # Encoding / Codec / QualitySpec
│   │       ├── codec/           # 誤差上限付きコーデック (sz / zfp feature)
│   │       └── error.rs
│   ├── aex-wire/                 # データプレーンのフレーム定義 (server/client 共用)
│   │   └── src/
│   │       ├── frame.rs          # FrameHeader のエンコード/デコード
│   │       ├── handshake.rs      # HELLO / READY
│   │       └── scatter.rs        # ScatterBuffer (並列受信用の非重複バッファ分割)
│   ├── aex-proto/                # tonic-build による生成コード
│   ├── aex-server/
│   │   └── src/
│   │       ├── main.rs
│   │       ├── control.rs        # tonic サービス実装
│   │       ├── session.rs        # SessionRegistry / FileRegistry
│   │       ├── transfer.rs       # TransferRegistry (plan の保持と TTL)
│   │       └── dataplane.rs      # TCP accept ループと送信スレッド
│   ├── aex-client/               # Rust クライアント (Python から独立して使える)
│   │   └── src/
│   │       ├── client.rs
│   │       ├── pool.rs           # DataPool
│   │       ├── fetch.rs          # チャンクスケジューラ
│   │       └── proxy.rs
│   └── aex-py/                   # PyO3 バインディング (cdylib: _aex)
│       └── src/lib.rs
├── python/
│   └── aex/
│       ├── __init__.py
│       ├── client.py             # Client / FileProxy / GroupProxy
│       ├── array_proxy.py        # ArrayProxy (__array_function__ 等)
│       └── py.typed
├── benchmarks/
│   ├── create_benchmark_data.py  # v1 から移植
│   └── run_benchmarks.py         # v1/v2 を同一条件で比較
└── tests/
    ├── rust/                     # 統合テスト
    └── python/                   # v1 の pytest を移植
```

### 4.1 クレート分割の理由

- `aex-core`: サーバ専用ロジック (バックエンド、選択解決、集約) を tokio/tonic から独立させ、単体テストを高速に回せるようにする
- `aex-wire`: ワイヤフォーマットをサーバとクライアントで**同一コードから**生成する。フォーマット不整合はプロトコル実装の最大のバグ源であり、定義を共有することで構造的に防ぐ
- `aex-client` を PyO3 から分離: Rust だけのクライアントとしても使え、ベンチマークで Python 層のオーバーヘッドを切り分けられる

---

## 5. コントロールプレーン仕様 (gRPC)

v1 の `aex.proto` を土台に、**データ転送 RPC を「転送計画の発行」に置き換える**。ファイル操作系はほぼそのまま維持する。

コントロールプレーンの TCP 接続は、サーバ・クライアントとも **`TCP_NODELAY` を必ず有効にする**。ここを流れるのは小さい要求と MSS に満たない応答の往復ばかりで、Nagle が最後の断片を握ると相手の遅延 ACK (40 ms) まで出ない。inline 返却される小さい選択 (§5.6.2) はこの 1 往復がそのまま応答時間になるため、影響が最も大きい。gRPC ライブラリに listener や connector を自前で渡す場合は、ライブラリ側の TCP 設定が適用されない点に注意する。データプレーンも同じ理由で常時有効とし、どちらも設定項目にしない。

### 5.1 サービス定義

```proto
syntax = "proto3";
package aex.v2;

service AexControl {
    // --- セッション管理 (v2 で新規) ---
    // データプレーンのエンドポイントとセッショントークンを取得する。
    rpc Connect(ConnectRequest) returns (ConnectReply);
    rpc Disconnect(DisconnectRequest) returns (DisconnectReply);

    // --- ファイル操作 (v1 から継承) ---
    rpc OpenFile(OpenFileRequest) returns (OpenFileReply);
    rpc CloseFile(CloseFileRequest) returns (CloseFileReply);
    rpc GetItem(GetItemRequest) returns (Item);
    rpc ListChildren(ListChildrenRequest) returns (ItemList);

    // --- データ転送 (v2 で刷新) ---
    // 選択を解決し、転送計画を返す。小さい選択はデータ自体も inline で返す。
    rpc PrepareSelection(PrepareSelectionRequest) returns (TransferPlan);
    // 複数の選択を 1 往復でまとめて解決する (gather、§10.3)。
    rpc PrepareSelections(PrepareSelectionsRequest) returns (TransferPlanList);
    // 転送の明示的な解放 RPC は持たない。plan は TTL と LRU で回収される (§5.6.1)。

    // --- サーバサイド計算 ---
    // 集約結果は小さいため unary で inline 返却する。
    rpc ApplyFunction(ApplyFunctionRequest) returns (ApplyFunctionReply);
}
```

**v1 からの変更点**

| v1 | v2 | 理由 |
|----|----|------|
| `GetSelection` → `stream Selection` | `PrepareSelection` → `TransferPlan` | データを gRPC から外し、生 TCP へ移す |
| `ApplyFunction` → `stream Selection` | `ApplyFunction` → `ApplyFunctionReply` (unary) | 集約結果は数バイト〜数 KB。ストリームは過剰 |
| セッション概念なし | `Connect` / `Disconnect` | データプレーンの認証とプール確立に必要 |
| ファイルハンドル = UUID 文字列 | `uint64` | 文字列比較とアロケーションを避ける |

### 5.2 セッション確立

```proto
message ConnectRequest {
    uint32 protocol_version   = 1;  // データプレーンのフレームバージョン (初版 = 1)
    uint32 desired_streams    = 2;  // クライアントが張りたいデータ接続数 (0 = サーバ既定)
    string client_name        = 3;  // ログ用
}

message DataEndpoint {
    string host = 1;   // 空文字ならコントロールプレーンと同一ホスト
    uint32 port = 2;
}

message ConnectReply {
    bytes  session_id             = 1;  // 16 バイト
    bytes  session_token          = 2;  // 16 バイト。データ接続の認証に用いる
    repeated DataEndpoint endpoints = 3;  // 初版は 1 個。将来の複数サーバ分散の枠
    uint32 granted_streams        = 4;  // サーバが許可した接続数
    uint32 protocol_version       = 5;
    uint64 default_chunk_bytes    = 6;  // サーバ推奨のチャンクサイズ
    uint64 max_fetch_bytes        = 9;  // 1 回の FETCH で要求してよい上限 (セッション中一定)
    uint32 supported_codecs       = 7;  // ビットマスク (bit0=RAW, bit1=SZ, bit2=ZFP, bit3=GZIP)
    uint32 supported_encodings    = 8;  // ビットマスク (bit0=EXACT, bit1=CAST, bit2=ERROR_BOUND)
}
```

`endpoints` を複数返せる形にしてあるのは、研究テーマである「広域拠点に分散したデータ」への拡張余地のためである。初版はサーバ自身の 1 エンドポイントのみを返す。

### 5.3 ファイル操作とメタデータ

v1 とほぼ同一。dtype 列挙も v1 の 14 種をそのまま維持する。

```proto
enum DataType {
    INT8 = 0; INT16 = 1; INT32 = 2; INT64 = 3;
    UINT8 = 4; UINT16 = 5; UINT32 = 6; UINT64 = 7;
    FLOAT16 = 8; FLOAT32 = 9; FLOAT64 = 10;
    COMPLEX64 = 11; COMPLEX128 = 12; BOOL = 13;
}

message Dataset { DataType dtype = 1; reserved 2; repeated int64 shape = 3; }  // 2 は旧 ndim (shape の長さと重複)
message Group   {}

message AttrArray { DataType dtype = 1; repeated int64 shape = 2; bytes data = 3; }
message Attribute { string name = 1; oneof value { string text = 2; AttrArray array = 3; } }

message Item {
    string name = 1;
    oneof data { Dataset dataset = 2; Group group = 3; }
    repeated Attribute attrs = 4;  // 名前順
}

message OpenFileRequest  { bytes session_id = 1; string path = 2; string format = 3; }
message OpenFileReply    { uint64 handle = 1; }
message CloseFileRequest { bytes session_id = 1; uint64 handle = 2; }
message GetItemRequest   { bytes session_id = 1; uint64 handle = 2; string name = 3; }
message ListChildrenRequest { bytes session_id = 1; uint64 handle = 2; string name = 3; }
message ItemList         { repeated Item items = 1; }
```

**属性** はグループにもデータセットにも付くので `Item` に持たせる。`map` ではなく `repeated` なのは、`map` に順序が無いためである。数値は dtype と shape を付けて生バイトで運ぶ。`double` に丸めると `_FillValue` が変数と違う型で届き、CF の読み手が使えなくなる。表現できない型 (compound、enum、参照、文字列の配列) の属性は一覧から落とす (第 7.5 節)。

**`.npy` の階層表現** は v1 と互換を保つ。ルートグループ `/` の下に単一データセット `array` が存在する。

### 5.4 選択の指定

v1 の `Index` に **`Ellipsis` と `NewAxis` を追加**する。v1 の `__getitem__` はこれらを扱えず、`arr[..., 0]` や `arr[:, None]` が失敗する。

```proto
message Slice {
    optional int64 start = 1;
    optional int64 stop  = 2;
    optional int64 step  = 3;
}
message Fancy { repeated int64 indices = 1; }

message Index {
    oneof kind {
        int64  single   = 1;   // arr[5]
        Slice  slice    = 2;   // arr[10:20:2]
        Fancy  fancy    = 3;   // arr[[1, 5, 10]]
        bool   ellipsis = 4;   // arr[...]
        bool   newaxis  = 5;   // arr[None]
    }
    reserved 6;   // 旧 mask_true_indices。boolean mask はクライアントが展開して fancy で送る
}
```

boolean mask はクライアント側で `np.nonzero` により整数インデックスへ展開してから送る。mask 自体を送ると、選択が疎な場合に mask のほうが選択結果より大きくなりうるため。

**インデックス数の上限**: 展開後の `Fancy.indices` は `repeated int64` であり、要素数に比例してリクエストが大きくなる。100 万要素で 8 MB となり、gRPC の既定メッセージ上限 (4 MiB) を超える。サーバは `max_fancy_indices` (既定 262144 = リクエスト約 2 MB) を超える選択を**エラーとして拒否**し、クライアントは同じ上限を持って送信前に検査する (無駄な往復を避けるため)。

エラーメッセージには、選択を分割するか `read_into` で範囲取得に置き換えるよう促す旨を含める。より大きい選択を効率的に送る形式 (mask のビットマップ表現、差分 + varint 符号化、サーバ側での述語評価) は将来課題とする (§14.1)。

### 5.5 適応品質の指定

```proto
enum Encoding {
    EXACT      = 0;  // 無損失。どのビルドも必ず出せる
    DTYPE_CAST = 1;  // float64 -> float32、float32 -> float16 (§5.5.3)
    ERROR_BOUND = 2; // 誤差上限付き非可逆圧縮。SZ3 か ZFP が運ぶ (§5.5.1)
}

message QualitySpec {
    Encoding encoding = 1;
    optional DataType cast_dtype = 2;          // DTYPE_CAST 用
    optional double abs_error_bound = 3;       // ERROR_BOUND 用
    optional double rel_error_bound = 4;
}
```

**ネゴシエーションの規約**: クライアントは希望する `QualitySpec` を送る。サーバは対応できない場合、**エラーにせず `EXACT` にフォールバック**し、実際に適用した内容を `TransferPlan.applied_quality` に入れて返す。クライアントは `applied_quality` を見て、返ってきた配列の実際の shape / dtype を決定する。

この「要求と適用の分離」により、出せるものが違うサーバとクライアントが混在しても壊れない。`EXACT` しか出せないサーバ (コーデックを 1 つも入れていないビルド) と `ERROR_BOUND` を求めるクライアントの組み合わせは、実際に起こりうる。同じことがコーデックの選択にも当てはまる: ZFP を求められた SZ3 だけのサーバは、黙って SZ3 で送らず `EXACT` に落とす。誤差の性質はコーデックごとに違うので、「誤差上限付きなら何でもよい」とは限らないからである。

#### 5.5.1 ERROR_BOUND の実装

`abs_error_bound` を適用できるのは、誤差上限付きコーデックを入れたビルドである。圧縮アルゴリズムは自前で実装せず外部ライブラリに委ねる。現在 2 つあり、どちらも既定 OFF である。

| `Codec` | feature | ライブラリ | 性質 |
|---|---|---|---|
| `SZ` (1) | `sz` | `sz3` crate (SZ3 を vendor し cmake + C++17 でビルド) | 予測と量子化。誤差上限を与えた数値のまま守る |
| `ZFP` (2) | `zfp` | `zfp-sys` crate (LLNL の zfp を vendor し cmake でビルド) | ブロック変換。誤差上限は**2 の冪に切り下げて**守る。**NaN / Inf を含むブロックは壊れる** |

適用するのは float32 / float64 のみで、それ以外の dtype と `rel_error_bound` は EXACT にフォールバックする。

**どちらを使うかは転送ごとにクライアントが選ぶ**。`PrepareSelectionRequest.requested_codec` に入れ、サーバが実際に使ったものが `TransferPlan.codec` で返る。この 2 つのフィールドはもともと可逆コーデック用に取ってあったもので、誤差上限付きのコーデックが 2 つになった時点で意味を持った。指名を省いたとき (および `RAW` のような誤差上限付きでない値が入っていたとき) は、サーバが自分の既定を使う。古いクライアントはこのフィールドを 0 のまま送るので、0 を「指定なし」と読むことが互換性の条件である。

ZFP が誤差上限を 2 の冪に切り下げることは、圧縮率を比べるときに効く。`abs_error=1e-1` を求めると ZFP が実際に守るのは 2⁻⁴ = 6.25e-2 なので、同じ数値で SZ3 と並べた圧縮率は ZFP に不利な側に寄っている。

非有限値については、**ZFP を選んだ転送だけ誤差上限の保証が外れる**。zfp のブロックに NaN や Inf が 1 つ入ると、そのブロックの他の値も上限をはるかに超えて壊れる。走査して弾くことはしない: 全ブロックを 1 回なめるコストを、zfp 自身の文書が「渡すな」と言っているデータのために払うことになるからである。SZ3 にこの制限は無い。

`rel_error_bound` を適用しないのは意図的である。値域に対する相対誤差は「選択全体の値域」を意味しなければ保証にならないが、圧縮ブロックは自分の範囲の値域しか見ない。ブロックごとに基準が変わる保証は保証ではないため、意味を確定できるまでフォールバックさせる。

**ブロック**: 1 つの `DATA` フレームのペイロードが、独立した 1 ブロックである。ブロック単体で展開できるので、順不同の到着・接続間のワークスティーリング・失われたチャンクの再取得が、無損失のときとまったく同じに成立する。ペイロードの先頭 24 バイトは AEX 自身のブロックヘッダで、要素型・ブロックの形状 (高々 3 軸)・誤差上限を持つ。受信側は転送の文脈を何も持たずに展開でき、ZFP のようにストリームが形状を含まないコーデックも同じ経路に載る (実際 ZFP は自分のストリームヘッダを一切書かない)。

**形状を渡すことが本質**: 論理バイト列は出力配列の C 順なので、行境界に揃ったバイト区間は「連続した行の集合」、すなわち最も遅い軸だけが部分的な N 次元スラブである。誤差上限付き圧縮器はどれも全軸の近傍から予測するため、平坦な要素列として渡すと本来の能力の多くを捨てることになる。同一の 256 MiB の float32 を端から端まで転送した実測で、誤差上限 1e-3 のときの圧縮率は 32768 × 2048 と伝えた場合が 9.70 倍、同じバイト列を 1 次元と伝えた場合が 5.85 倍だった ([docs/benchmark-sz.md](docs/benchmark-sz.md))。

圧縮した転送は、帯域がコーデックの速度を上回っているあいだは CPU で律速される。論理スループットは `min(コーデックの CPU の天井, 帯域 × 圧縮率)` でよく説明でき、実測の 10 点を数 % 以内で予言する ([docs/benchmark-zfp.md](docs/benchmark-zfp.md))。**どちらの項が効くかがコーデックの選び方そのものである**: 狭い回線では圧縮率が、広い回線では CPU の天井が、それぞれ単独で効く。

遅延を足すとどちらのコーデックも目減りするが (RTT 50 ms で SZ3 −15 %、ZFP −42 %)、仕組みは分かっていない。当初は「credit が論理バイト建てなので r 倍に圧縮すると線に乗る in-flight が r 分の 1 になる」と考えていたが、それなら圧縮率の高い SZ3 のほうが大きく落ちるはずで、向きが逆である。credit を 4 倍にすると両方とも更に遅くなるので、窓が小さすぎるのでもない。

コーデックは接続スレッドの中で同期に走る。外部ライブラリ側のスレッド化 (SZ3 の OpenMP、zfp の OpenMP) は使わない。zfp は静的リンクすることで OpenMP ごと落としている。使うと 1 接続が自分のスレッドチームを張り、接続数 × チーム数でコアを奪い合うことになる。並列度は接続数だけで決まる、という単純な関係を保つ。その代わり `streams` が圧縮の並列度そのものになるので、圧縮転送では無損失転送と最適値が違う。

そのためサーバは、誤差上限付きの転送に限り読みピースを行バイト数の倍数に丸める (§6.5.2)。行がピースより長い配列では要素単位で切り、そのブロックは 1 次元として扱う。揃わなくても正しさは変わらず圧縮率が落ちるだけなので、クライアントに整列の義務は課さない。ただしブロックは要素を分割できないため、要素境界に揃っていない `FETCH` だけは `REQUEST` エラーで拒否する。

**フレーム**: 圧縮したブロックが元より大きくなる場合 (SZ3 は数十バイトの自前ヘッダを持ち、ZFP は部分ブロックを 4 要素ぶんに詰めるため、要素数の少ないブロックでは必ず起きる) は、そのフレームだけ `RAW` で送る。`codec` はフレームごとの値なので、1 つの転送が両方を運んでもよい。

#### 5.5.2 GZIP

無損失の `Codec` は `GZIP` (3) ひとつだけで、`flate2` (zlib-rs、pure Rust) の deflate を
レベル 6 で使う。外部ツールチェーンが要らないので**既定 ON** である。

`EXACT` と組む唯一のコーデックであり、ここで初めて `encoding` と `codec` が直交する。
それ以前は `EXACT ⟺ RAW` の 1 対 1 だった。ブロックヘッダは他のコーデックと同じものを
前に置く。gzip は形状も要素型も見ないが、受信側が全コーデックで同じ検査 (ブロックが
出力の何バイトに展開されるか) を通せるほうが安い。誤差上限を持たないので、ヘッダの
ε には 0 が入る。

**これは比較対象であって、選ぶべきものではない。** 実データの float32 に対する圧縮率は
1.1〜1.4 倍で、同じデータに誤差上限付きコーデックを掛ければ人が見て分からない誤差で
桁違いに縮む。既定には決してしない。指名されたときだけ使う。

無損失コーデックの要求は、**出せなければ黙って `RAW` に落ちる**。誤差上限付きコーデックの
指名と違い、届くバイト列は同じなので、落ちたことを品質として報告する意味がないからである。

#### 5.5.3 DTYPE_CAST

許すのは **float64 → float32** と **float32 → float16** の 2 つだけである。どちらも要素幅が
ちょうど半分になり、数の意味は変わらない。float → int のように数の種類が変わるキャストは、
飽和・丸め・符号をすべて決める別の問題なので、ここでは答えない。float64 → float16 も
通さない: 2 段飛ばしの精度損失を 1 回の要求で起こす理由がない。

**これは他のどの品質とも性質が違う。** 誤差上限付き圧縮と違って、
`wire_len == logical_len` が保たれる — 論理ストリームそのものが半分になるだけで、
コーデックは `RAW` のままである。結果として:

- 転送前に総バイト数が確定する (`TransferPlan.total_bytes` がちょうど半分)
- 受信側は出力配列に直接読む。**展開のコストがゼロ**
- 「圧縮したら大きくなったので RAW で送る」のフレームごとの分岐が起きない
- credit が論理バイト建てであることが問題にならない
- クライアントが確保する配列も半分になる。誤差上限付き圧縮は float32 のまま返すので減らない

**値が収まるかは検査しない。** float16 は 65504 を超えると `inf` になる。
`numpy.astype` と同じ振る舞いで、サーバは走査しない。収まるかを知るには合意の前に
選択全体を読む必要があり、それは転送そのものより高くつく。`TransferPlan.applied_quality`
は `DTYPE_CAST` と報告し、届いた値を見れば分かる。

**変換の位置**: `SelectionLayout::read_with` の中、backend より上である。backend は
自分のストレージだけを知っていればよいという性質を保つため、キャストを知っているのは
レイアウトだけにしてある。レイアウトは dtype を 2 つ持つ: `dtype` が線に乗る型、
`src_dtype` が配列自身の型で、ソースに届く計算はすべて 2 つの比で換算する。gather の
walk は要素単位なので、ソース要素で読んでから narrow する順になる。

### 5.6 転送計画

```proto
message PrepareSelectionRequest {
    bytes  session_id = 1;
    uint64 handle     = 2;
    string name       = 3;
    repeated Index indices = 4;
    QualitySpec requested_quality = 5;
    uint32 requested_codec = 6;       // どのコーデックか。ERROR_BOUND では誤差上限付きのどれか。0 = RAW
}

message TransferPlan {
    uint32 request_id   = 1;   // データプレーンで使う識別子 (セッション内で一意)
    bytes  ticket       = 2;   // 16 バイト。この転送に対する capability
    DataType dtype      = 3;   // 実際に転送される dtype
    repeated int64 shape = 4;  // 実際に転送される shape
    uint64 total_bytes  = 5;   // 論理バイト列の総長 (非圧縮)
    uint32 codec        = 6;   // 実際に使う codec
    QualitySpec applied_quality = 7;

    reserved 8;   // 旧 expires_unix_ms。読むクライアントがなかった。期限は §6.9 のとおりサーバが管理する

    // total_bytes <= inline_limit_bytes のとき、データ本体をここに入れて返す。
    // データプレーンを使わずに 1 RTT で完結させるための経路 (§5.6.2)。
    bytes inline_data = 9;
}

message PrepareSelectionsRequest {
    bytes session_id = 1;
    repeated PrepareSelectionRequest requests = 2;   // 各要素の session_id は無視される
}

message TransferPlanList {
    // requests と同順・同数。個別に失敗した要素は error を持ち、plan は空になる。
    repeated TransferPlanOrError results = 1;
}

message TransferPlanOrError {
    oneof result {
        TransferPlan plan  = 1;
        PlanError    error = 2;
    }
}

// データプレーンの ERROR フレームおよび READY の status と同じ値を使う (§6.3)。
enum ErrorClass {
    OK        = 0;
    PROTOCOL  = 1;   // フレーム / ハンドシェイクの構文違反
    AUTH      = 2;   // セッション・トークン・チケットの不正または失効
    PLAN      = 3;   // plan が存在しない、または期限切れ
    REQUEST   = 4;   // 要求内容が不正 (範囲外、上限超過、未対応フォーマット)
    TRANSIENT = 5;   // サーバ側の一時的事情 (I/O エラー、リソース逼迫)
    PERMANENT = 6;   // サーバ側の恒久的事情 (ファイル破損、内部エラー)
}

message PlanError {
    ErrorClass klass = 1;
    string     message = 2;   // 人間向けの診断。機械判定に使ってはならない
}
```

**コントロールプレーンのエラーは gRPC の標準 `Status` をそのまま使う。** `ErrorClass` は 6 クラスとも標準ステータスコードに 1 対 1 で対応するため (下表)、独自のエラーコードを metadata などで重ねて運ぶ必要はない。`ErrorClass` を明示的に使うのは `PrepareSelections` の要素別エラーだけである。これは 1 つの RPC に `Status` を 1 つしか付けられず、N 個の選択のうち一部だけが失敗する状況を表現できないためである。

| `ErrorClass` | gRPC `Code` |
|---|---|
| `PROTOCOL` | `INTERNAL` |
| `AUTH` | `UNAUTHENTICATED` |
| `PLAN` | `FAILED_PRECONDITION` |
| `REQUEST` | `INVALID_ARGUMENT` / `NOT_FOUND` / `PERMISSION_DENIED` |
| `TRANSIENT` | `UNAVAILABLE` |
| `PERMANENT` | `DATA_LOSS` |

`PrepareSelections` は `gather` (§10.3) の基盤である。N 個の選択を個別の `PrepareSelection` で解決すると N × RTT かかり、`gather` の目的である RTT 削減が達成できない。**1 要素の失敗が全体を失敗させない**ようにするため、結果は要素ごとに plan かエラーかを持つ形とする (numpy の選択は片方だけが範囲外になることが普通にあるため)。

#### 5.6.1 plan のライフサイクルと明示解放の不在

転送の完了をサーバに通知する RPC は持たない。plan は以下の 3 つで回収される。

1. **TTL 経過** — `FETCH` を受け取るたびに `transfer_ttl_sec` 先へ延長される (§6.9)。したがって期限切れは「クライアントが実際に沈黙した場合」にのみ起きる
2. **LRU 破棄** — セッションの plan 数が `max_transfers_per_session` に達したら、最も古く参照された plan から破棄する
3. **セッション終了**

明示解放 RPC を置かない理由は、それが**正しさに寄与しないため**である。解放をクリティカルパスに入れれば 1 往復ぶんレイテンシが増え (§5.6.2)、クリティカルパス外に出せば「少し早く解放できる」だけの最適化になる。上の 3 つで plan 数もメモリも上から縛れており、早期解放の価値はそれに見合わない。

plan エントリ自体は `Arc<dyn ArrayDataset>` と `SelectionLayout` のみで小さく、`max_transfers_per_session` (既定 64) で個数も縛られている。inline 返却 (次節) により小さい選択は plan を作らないため、対話的な利用で plan が溜まることもない。

#### 5.6.2 小さい選択の inline 返却

`total_bytes` が `inline_limit_bytes` (既定 64 KiB) 以下のとき、サーバは `TransferPlan.inline_data` にデータ本体を入れて返し、`request_id` と `ticket` は発行しない (ゼロ埋め)。クライアントはデータプレーンを一切使わず、その場で転送を完了させる。

これはプレーン分離の構造的なコストを埋めるために必須の経路である。データプレーンを必ず経由する設計では、1 回の小さい選択に

```
PrepareSelection (1 RTT) + FETCH/DATA (1 RTT) = 2 RTT
```

かかり、`GetSelection` が 1 RTT で済んでいた v1 より**遅くなる**。第 1.2 節で律速要因 #6 として挙げた「対話的分析での RTT の積算」は本基盤の主要な問題意識であり、大きい連続転送を速くする代償に小さい選択を遅くするのでは本末転倒である。inline 経路によりこれは 1 RTT になり、v1 と同等になる。

| 選択サイズ | 往復数 | 経路 |
|-----------|-------|------|
| `<= inline_limit_bytes` | 1 RTT | `PrepareSelection` のみ |
| `> inline_limit_bytes` | 2 RTT | `PrepareSelection` + データプレーン |

**閾値を 64 KiB に抑える理由**: inline 経路のデータは protobuf のエンコード/デコードを通るため、第 1.3 節で排除したユーザ空間コピーが復活する。RTT 削減という目的は数十 KiB で達成できるので、閾値を上げてコピーの当たる範囲を広げる意味はない。1 行の取得 (数 KB) や小さいタイルの取得といった、レイテンシが支配的な用途だけを拾う値とする。

**論理バイト列 (logical byte stream)** が本設計の中心概念である。転送対象の配列を C 順で平坦化した `total_bytes` バイトの仮想的な連続バイト列を定義し、すべてのチャンク・フレーム・オフセットはこの上の座標で表す。

この定義により:
- クライアントは `np.empty(shape, dtype)` を 1 個確保するだけでよく、offset がそのままバッファ内位置になる
- チャンクを**どの接続で受け取っても**、どの順序で届いても、正しい位置に書ける
- 接続が切れたチャンクだけを再フェッチできる (冪等性)

**チャンク分割はサーバではなくクライアントが行う** (第 6.4 節)。`TransferPlan` は論理バイト列の長さと上限だけを伝え、どう切るかには関与しない。理由は以下の通り。

- 最適なチャンクサイズを決める情報 (接続数、実測 RTT・帯域、credit) は**すべてクライアント側にある**。credit とチャンクサイズは `credit = BDP / chunk_bytes` で連動するため、片方をサーバが決めると噛み合わない
- チャンク数が接続数を下回ると接続が遊ぶ。この判断はクライアントにしかできない
- サーバがチャンクリストを protobuf で返す必要がなくなる (100 GB を 4 MiB で切ると 25600 要素になっていた)
- 圧縮バックエンドでの伸長重複は、境界を揃えるのではなく**サーバ側のデコードキャッシュ**で解決する (第 7.4 節)。したがってサーバがストレージ境界をクライアントに伝える必要もない

サーバ側の制約は `max_fetch_bytes` (1 回の `FETCH` の上限) のみで表現する。これも既定チャンクサイズの推奨と同じくサーバ設定由来でセッション中ずっと同じ値なので、転送ごとではなく `ConnectReply` で一度だけ伝える。

### 5.7 サーバサイド計算

```proto
message FunctionArgument {
    oneof value {
        int64    int_value   = 1;
        double   float_value = 2;
        bool     bool_value  = 3;
        bool     none_value  = 4;
        IntTuple tuple_int   = 5;
    }
}
message IntTuple { repeated int64 values = 1; }

message ApplyFunctionRequest {
    bytes  session_id = 1;
    uint64 handle     = 2;
    string name       = 3;
    string function_name = 4;
    map<string, FunctionArgument> kwargs = 5;
    repeated Index indices = 6;   // v2 追加: 部分配列に対する集約を可能にする
}

message ApplyFunctionReply {
    DataType dtype = 1;
    repeated int64 shape = 2;
    bytes data = 3;                   // 結果が inline_limit 以下ならここに入る
    reserved 4;                       // 旧 plan (超える場合のデータプレーン経由)。一度も送られなかった
}
```

v1 は `ApplyFunction` が常に**データセット全体**を読んでから関数を適用していた (`dataset.get_full_array()`)。v2 では `indices` を受け取り、選択に対して集約できるようにする。`np.sum(arr[0:100])` が 100 行だけ読んで済む。

`inline_limit` は §8.2 の `inline_limit_bytes` (既定 64 KiB) を共有する。初版で実装する集約関数は結果が小さいものに限るため、`plan` 経路は使われない (フィールドのみ確保)。

### 5.8 実装する集約関数

| 分類 | 関数 |
|------|------|
| 総和・積 | `sum`, `prod`, `mean` |
| 最大最小 | `max`, `min` |
| 分散 | `std`, `var` (`ddof` 対応) |
| 論理 | `all`, `any` |
| インデックス | `argmax`, `argmin` |
| NaN 対応 | `nansum`, `nanmean`, `nanmax`, `nanmin` |

いずれも `axis` (int / tuple / None) と `keepdims` に対応する。`median` / `sort` / `argsort` / `cumsum` 系は初版では**未実装**とし、クライアントは警告付きでローカルフォールバックする。

**numpy とのセマンティクス一致**は差分テスト (第 11 章) で保証する。特に以下に注意する。

- 空配列に対する `max`/`min` はエラー、`sum` は 0、`all` は True
- `mean`/`std`/`var` の整数入力は float64 で計算する
- `argmax`/`argmin` は C 順の平坦化インデックスを返す (axis=None の場合)
- NaN 系は全要素が NaN のスライスで警告と NaN を返す

---

## 6. データプレーン仕様 (生 TCP)

本章が AEX2 の中核である。

### 6.1 設計原則

1. **ペイロードは一切変換しない** — ヘッダとペイロードは明確に分離され、ペイロードは論理バイト列そのものである。これにより受信側は `read_exact` で出力配列へ直接読める (ゼロコピー受信)
2. **固定長ヘッダ** — 可変長ヘッダは 2 段読みか先読みバッファを要求し、ゼロコピー受信を壊す。32 バイト固定とする
3. **すべてのオフセットは論理バイト列座標** — 受信バッファ上の位置が計算不要で決まる
4. **接続は状態を持たない (ハンドシェイク後)** — どの接続でどのチャンクを運んでもよい。これが並列化・ワークスティーリング・再送の自由度を生む
5. **フレームは自己完結** — フレーム単体で「どの転送の、どの位置の、何バイトか」が分かる

### 6.2 接続確立 (ハンドシェイク)

クライアントは `ConnectReply.granted_streams` 本の TCP 接続を張る。各接続で 1 回だけハンドシェイクを行う。

**HELLO (client → server), 48 バイト固定**

| offset | size | field | 内容 |
|--------|------|-------|------|
| 0 | 8 | magic | ASCII `"AEXDATA\x01"` |
| 8 | 2 | version | プロトコルバージョン (u16 LE) |
| 10 | 2 | reserved | 予約 (0) |
| 12 | 4 | flags | 予約 (0) |
| 16 | 16 | session_id | `ConnectReply.session_id` |
| 32 | 16 | session_token | `ConnectReply.session_token` |

**READY (server → client), 16 バイト固定**

| offset | size | field | 内容 |
|--------|------|-------|------|
| 0 | 8 | magic | ASCII `"AEXDATA\x01"` |
| 8 | 2 | status | `ErrorClass` (0 = OK)。§6.3 |
| 10 | 2 | version | サーバが採用したバージョン |
| 12 | 4 | flags | 予約 |

`status != 0` の場合、サーバは READY 送出後ただちに接続を閉じる。値は §6.3 の `ErrorClass` であり、magic やバージョンの不一致は `PROTOCOL`、セッション・トークンの不正は `AUTH`、接続数超過は `TRANSIENT` となる。より詳細な理由はサーバのログに残す。READY は 16 バイト固定でメッセージを運べないが、クライアントが取るべき動作はクラスだけで決まるため不足はない。

**認証モデル**: `session_token` は `Connect` RPC の応答としてのみ配布される 128 ビットの乱数である。データプレーンはこれを検証するだけで、ユーザ認証やアクセス制御はコントロールプレーン (gRPC) 側の責務とする。転送単位の `ticket` は各 `FETCH` フレームのペイロードとして提示され (§6.3)、`session_token` に加えてチェックされる。これにより他セッションの転送を横取りできないことを保証する。

平文であるため、盗聴・改竄への耐性はない。これは「少数クライアント・信頼できる環境」という前提に基づく初版の割り切りであり、`flags` に TLS ネゴシエーション用のビットを予約してある。

### 6.3 フレームヘッダ (32 バイト固定)

すべての多重化フレームは以下のヘッダで始まる。整数はすべてリトルエンディアン。

| offset | size | field | 説明 |
|--------|------|-------|------|
| 0 | 1 | `frame_type` | フレーム種別 (下表) |
| 1 | 1 | `codec` | 0=RAW, 1=SZ, 2=ZFP, 3=GZIP |
| 2 | 1 | `encoding` | `QualitySpec.Encoding` と同じ値 |
| 3 | 1 | `flags` | 予約 (0)。将来の拡張用 |
| 4 | 4 | `request_id` | `TransferPlan.request_id` (u32) |
| 8 | 8 | `offset` | 論理バイト列上の開始オフセット (u64) |
| 16 | 8 | `wire_len` | 続くペイロードのワイヤ上の長さ (u64) |
| 24 | 8 | `logical_len` | ペイロードの論理長 = 展開後の長さ (u64) |

`codec = RAW` のとき `wire_len == logical_len` であり、このときのみゼロコピー受信が成立する。圧縮されたフレームは `wire_len < logical_len` となり、受信側は一時バッファへ読んでから展開する (ゼロコピーは諦める)。codec は**フレームごと**の値なので、1 つの転送が圧縮フレームと RAW フレームを混ぜて運んでよい (§5.5.1)。

**フレーム種別**

| 値 | 名前 | 方向 | ペイロード | 説明 |
|----|------|------|-----------|------|
| 0x01 | `FETCH` | C→S | 16 バイト (ticket) | `offset` から `logical_len` バイトを要求する。`wire_len` は 16 |
| 0x02 | `DATA` | S→C | あり | 実データ |
| 0x03 | `ERROR` | S→C | あり (下記) | `FETCH` の処理に失敗した |
| 0x04 | (予約) | — | — | 未使用。将来の拡張用に空けてある |
| 0x05 | `PING` | 双方向 | なし | キープアライブ |
| 0x06 | `PONG` | 双方向 | なし | PING への応答 |

`FETCH` はチャンク境界に一致している必要はないが、クライアントは `TransferPlan` のチャンク分割に従うことを推奨する。サーバは任意のレンジ要求を処理できなければならない (再送時に部分レンジが発生するため)。

サーバは 1 つの `FETCH` に対して、`logical_len` を満たすまで 1 個以上の `DATA` フレームを返す。分割するかどうかはサーバの裁量である (例: 選択が非連続で、集約バッファのサイズに上限がある場合)。

**`ERROR` フレーム**: ペイロードは先頭 1 バイトの `ErrorClass` (§5.6 の proto enum と同じ値)、続いて UTF-8 の診断メッセージである (`wire_len = 1 + message.len()`)。ヘッダの `request_id` / `offset` / `logical_len` は**失敗した `FETCH` を指す**ため、本来の意味のまま使う。これがないと、後述する `TRANSIENT` の部分再試行ができない。セッションや接続全体のエラーには `request_id = 0` を用いる (`request_id` は 1 から採番する)。

```
ERROR フレーム
  ヘッダ 32 バイト
    request_id  = 失敗した転送 (接続全体のエラーなら 0)
    offset      = 失敗した FETCH の開始オフセット
    logical_len = 失敗した FETCH の長さ
    wire_len    = 1 + message.len()
  ペイロード
    [0]    u8   ErrorClass
    [1..]  UTF-8 の診断メッセージ
```

**エラーの分類はクラスのみで行い、個別のエラーコードは持たない。** クライアントの回復動作はクラスだけで決まり (§6.8)、同じクラス内で動作が分岐することはないためである。具体的な原因は診断メッセージとサーバログに残す。クラスを 6 個に固定することで、「サーバが新しいコードを足すと古いクライアントが分類できなくなる」という前方互換の問題も生じない。

**`FETCH` の ticket**: `FETCH` のペイロードは `TransferPlan.ticket` の 16 バイトである。サーバは `request_id` で plan を引いたうえで、(a) 提示された ticket が一致すること、(b) その plan が当該接続のセッションに属することの両方を検証し、いずれかが不一致なら `ERROR` を返して接続を閉じる。

固定長ヘッダに ticket を入れず、ペイロードとして送る形にしたのは、ヘッダを 32 バイトに保つためである。`FETCH` は C→S 方向でありゼロコピー受信の対象ではないため、2 段読み (ヘッダ 32 バイト → ペイロード 16 バイト) になっても性能上の不利はない。

この検証がないと、`request_id` がセッション内連番である以上、同一サーバに接続した別セッションが数回の総当たりで他人の転送を読み出せてしまう。**ticket の検証はデータプレーンにおける唯一のアクセス制御である。**

**転送完了の判定**: クライアントは自らチャンク分割を行う (§6.4) ため、全チャンクの完了を自分で追跡できる。したがってサーバが「最終フレーム」を通知する必要はなく、そのようなフラグも持たない。チャンクは複数接続へ動的に配られるので、どの接続が最後になるかはサーバには分からず、再送が起きれば「最後」は一意ですらない。

**中断フレームを持たない理由**: `DATA` は `FETCH` に対してのみ返されるため、転送を止めたければクライアントは `FETCH` の投入をやめるだけでよい。in-flight のデータは投入済み `FETCH` の分 (最大 `streams × credit × chunk_bytes`) に限られ、有限である。

同一接続は FIFO であるため、仮に中断フレームを送っても**先行する `FETCH` より後ろに並び、それらの処理を止められない**。サーバ側の plan は TTL と LRU で回収される (§5.6.1) ため、明示的に中断を伝える必要もない。したがって中断専用のフレームは定義しない。

**drain 規則**: 中断時、クライアントは各接続について「投入済み `FETCH` の `logical_len` の合計」に達するまで `DATA` を読み捨ててから、接続を次の転送に再利用してよい。`ERROR` を受け取った `FETCH` はその時点で完了とみなす。1 つの `FETCH` が複数の `DATA` に分割されても合計バイト数は変わらないため、この判定は常に可能であり、サーバからの完了通知 (ACK) を必要としない。**出力バッファ (`ScatterBuffer` が指す numpy 配列) の解放は、この drain 完了より前に行ってはならない** (第 6.6 節の寿命に関する不変条件)。

### 6.4 多重化とスケジューリング

```
Client                                              Server
  │                                                    │
  │ ── gRPC: PrepareSelection ───────────────────────→ │  選択を解決、plan を登録
  │ ←── TransferPlan{request_id=7, total=256MiB,       │
  │      ticket=...} ──────────────────────────────── │
  │                                                    │
  │  np.empty((...), dtype) を確保                      │
  │  chunk_bytes を 4MiB と決定 → 64 チャンクに分割     │
  │  8 本の接続へ動的に配布                             │
  │                                                    │
  │ ═S0═ FETCH{7, off=0,     len=4Mi} ───────────────→ │
  │ ═S0═ FETCH{7, off=32Mi,  len=4Mi} ───────────────→ │  (credit: 事前投入)
  │ ═S1═ FETCH{7, off=4Mi,   len=4Mi} ───────────────→ │
  │ ═S1═ FETCH{7, off=36Mi,  len=4Mi} ───────────────→ │
  │  ...                                               │
  │ ←═S0═ DATA{7, off=0, len=4Mi} + 4MiB raw ───────── │  pread → buf → writev
  │ ←═S1═ DATA{7, off=4Mi, len=4Mi} + 4MiB raw ─────── │
  │  各接続スレッドが out_buf[off..off+len] へ直接 read │
  │  ...                                               │
  │ ←═S3═ DATA{7, off=252Mi, len=4Mi} ──────────────── │
  │  全チャンク完了。plan は TTL / LRU で回収される      │
```

**チャンク分割**: `TransferPlan` を受け取ったクライアントが、論理バイト列を自ら分割する。サーバは関与しない (第 5.6 節)。

```
1. chunk_bytes = min(ConnectReply.default_chunk_bytes, ConnectReply.max_fetch_bytes)
2. チャンク数が接続数を下回るなら、全接続を使えるまで下げる:
     if ceil(total_bytes / chunk_bytes) < streams:
         chunk_bytes = max(MIN_CHUNK_BYTES, ceil(total_bytes / streams))
3. credit = clamp(ceil(BDP / chunk_bytes), 2, 32)     ← 6.4 の credit 式
4. chunks = ceil(total_bytes / chunk_bytes) 個に均等分割 (末尾のみ端数)
```

`MIN_CHUNK_BYTES` (既定 256 KiB) を下回らせないのは、フレームヘッダと syscall の割合が増えるため。**chunk_bytes を先に決め、credit をそれに従わせる**順序とする (`credit = BDP / chunk_bytes` なので逆順だと循環する)。

転送が小さく `chunk_bytes` が下限に張り付く場合はチャンク数が接続数を下回る。このときは使う接続を減らし、余った接続は起こさない。

**チャンク割り当て**: クライアントは未割り当てチャンクのキューを持ち、各接続スレッドが完了するたびに次を取る (**ワークスティーリング**)。静的ラウンドロビンではなく動的割り当てとする理由は、接続ごとに実効帯域が異なる場合 (WAN、経路差、NIC キューの偏り) に遅い接続が全体を律速しないようにするため。

**パイプライン深度 (credit)**: 各接続は「完了を待たずに投入してよい `FETCH` 数」を持つ。既定値は 4。WAN では RTT を隠蔽するためにこれが不可欠であり、以下の式で自動調整する。

```
credit = clamp( ceil( BDP / chunk_bytes ), 2, 32 )
BDP    = 推定帯域 × 実測 RTT
```

RTT はハンドシェイク往復と PING/PONG から、帯域は直近の転送実績から推定する。初回転送時は既定値を使う。

**チャンクサイズ**: 既定 4 MiB。M6 の掃引で 256 KiB / 1 / 4 / 16 MiB を測った結果、遅延のある回線では credit と等価に働く (どちらで in-flight バイト数を作っても同じ値になる) ため、4 MiB のまま credit で調整する ([docs/benchmark-m6.md](docs/benchmark-m6.md))。

**接続数**: 既定 8。M6 の掃引では、遅延のない回線で効くノブはこれだけだった。16 まで伸びるが、1 台のクライアントが `max_streams_per_session` (既定 32) の半分を取らないよう 8 とする。

**credit**: 既定 16。遅延のある回線でのスループットは `streams × credit × chunk_bytes` だけで決まる。16 は既定のチャンクサイズで 512 MiB を in-flight に置き、20 Gbit/s × 200 ms を覆う。

### 6.5 送信パス (サーバ)

`.npy` は C 順の連続領域であるため、選択結果の論理バイト列は**ソースファイル上の断片列**として表現できる。

```rust
/// 論理バイト列上の [logical_offset, logical_offset+len) が、
/// ソースファイル上の [src_offset, src_offset+len) に対応することを表す。
struct Fragment {
    logical_offset: u64,
    src_offset:     u64,
    len:            u64,
}
```

- **連続選択** (`arr[10:20]`, `arr[:]` など、最終軸がフル選択かつ step=1): 断片は 1 個
- **ストライド/部分選択** (`arr[:, 0:50]`): 断片は「行数」個。各断片は行内の連続領域
- **fancy indexing** (`arr[[5, 1, 9]]`): 断片は選択行数個。論理順序はインデックス列の順
- **要素単位に分解される選択** (`arr[:, ::2]`): 断片が要素数と同じ個数になる

断片列は遅延評価のイテレータとして表現し、実体化しない (`arr[:, ::2]` で断片が 10^9 個になっても、メモリを消費せずに順次生成できる)。

### 6.5.1 I/O 方式の選択

**`mmap` と `sendfile` は採用しない。読み出しは `pread`、送信は `writev` に統一する。**

ゼロコピー送信 (mmap + writev で 1 コピー、sendfile で 0 コピー) が成立するのは、**非圧縮フォーマットの無加工転送に限られる**。将来扱う Zarr / HDF5 は圧縮チャンクが前提でありデコードを経てバッファに載る。さらに本研究の核である適応品質転送 (dtype キャスト、誤差上限付き圧縮) を実装した時点で、加工経路が主となる。すなわちゼロコピー送信は「初版の `.npy` にしか効かず、将来消える最適化」であり、そのために `Source` の分岐・`IoSlice` のライフタイム管理・OS 分岐を恒久的に抱えるのは割に合わない。**バックエンドの内部事情を送信コードから完全に隠す**ことを優先する。副次的に、mmap の SIGBUS (転送中に truncate されるとプロセスが落ちる) と、ページフォルトがコールドキャッシュで I/O キューを深くできない問題も回避できる。

代償はサーバ送信側のユーザ空間コピー 1 回だが、10 GbE ではメモリ帯域に対して十分小さい (第 1.3 節)。**クライアント側のゼロコピー受信は維持される** (第 6.6 節)。高速リンクで必要になった場合、`sendfile` 経路は**フレームプロトコルを変えずに後から追加できる** (第 7.1 節)。

### 6.5.2 送信アルゴリズム

```
FETCH{offset, logical_len} を受け取る
  1. 断片イテレータを offset へシーク
       Contiguous / Strided は O(1)、Fragmented も解析的に計算
  2. 読みバッファ (接続ごとに確保して使い回す) へ読み出す
       → 断片ごとに pread してバッファへ順に詰める (下記)
       → デコードが必要なバックエンドは、ここで伸長結果を書く
  3. writev(header, &buf[..len]) で送出
       ヘッダ 32B とペイロードを 1 syscall にまとめる
```

**読み戦略**: ファイル上の非連続な断片を 1 つの連続バッファに集める操作 (gather read) を 1 回の syscall で行う手段は、POSIX には存在しない。`preadv` は「ファイルの連続領域を複数バッファへ散らす」scatter であり、向きが逆である。したがって断片ごとの `pread` ループが基本となるが、断片が細かい場合は syscall 回数が支配的になる。

**初版は断片ごとの `pread` ループのみとする。** 密な選択 (`arr[:, ::2]` など、断片が細かく間隔も狭い) では「断片群を包含する連続領域を 1 回の `pread` で読み、必要部分だけを集約する」ほうが速い可能性があるが、これは以下を抱え込む。

- over-read 用の一時バッファが必要になり、`read_range` が出力バッファだけを受け取る形 (§7.1) が崩れる。「転送中のアロケーションはゼロ」という性質も失われる
- over-read の暴走を防ぐ上限 (論理長比・絶対値) と、上限を超えた場合の分割ロジックが要る
- 切り替え閾値を決める根拠がまだない

いずれも**測定なしに払う複雑さ**である。断片ごとの `pread` で実際に syscall が律速するかどうかを M4 のベンチマーク (第 12 章) で確かめ、律速すると分かってから入れる (第 14.1 節)。その際もフレームプロトコルとトレイトの変更は不要で、`read_range` の内部実装だけで閉じる。

`writev` の `IOV_MAX` (Linux で 1024) は、ヘッダ + ペイロードの 2 要素しか使わないため問題にならない。

### 6.5.3 読みの先読み

`pread` はブロッキングであり、`write` との逐次実行ではディスクとネットワークが交互に遊ぶ。

```
逐次:    [pread][write][pread][write]              ← 片方が常に待つ
先読み:  [pread][write][pread][write]
           ↑ 次の piece の読みはカーネルが既に始めている
```

データが RAM に収まらない場合 (本基盤の本来の想定) は、この差が支配的になる。piece を読む前に、**その先 4 MiB ぶんを `posix_fadvise(WILLNEED)` でカーネルに読ませておく** (Linux のみ。他の OS では何もしない)。スレッドもバッファも増えず、接続あたり 1 スレッド・1 バッファのままである。

読みスレッドを別に立てて有界チャネルで繋ぐ案は測って採らなかった。バッファの受け渡しは 2 接続以上で損になり、1 接続でも**同じ 2 スレッドを 2 接続に使ったほうが速い** (`docs/benchmark-fadvise.md`)。

### 6.5.4 なぜ非同期 (tokio) ではなく専用 OS スレッドか

通常ファイルの読み出しは、Linux / macOS のいずれにも**真の非同期手段が存在しない**。`O_NONBLOCK` は通常ファイルの read に効かず、epoll / kqueue も常に ready を返す。`tokio::fs` が内部で `spawn_blocking` しているのはこのためである。したがって tokio を採用しても、ファイル読み出しは結局ブロッキングスレッドプールへ逃がすことになり、スレッドは減らない。

tokio の本来の利点は「読みと送りを重ねられる」ことだが、それは 6.5.3 の先読みで同等に達成できる。想定規模が「少数クライアント」であり、スレッド数は `クライアント数 × streams` (数十本程度) に収まるため、**接続ごとに専用 OS スレッド + 同期 I/O** が最も単純で、性能上の損もない。コントロールプレーンのみ tokio で動かす。

将来 `io_uring` を採用すれば `pread` / `send` の双方が真に非同期となる。Linux 専用のため初版では採らないが、**フレームプロトコルを変えずに差し替えられる**。

### 6.5.5 移植性

- `pread` は Rust std の `std::os::unix::fs::FileExt::read_at` で利用できる
- `writev` は `std::io::Write::write_vectored` で利用できる
- ベクタ版の pread (`preadv`) は std になく、かつ上記の通り断片読みには使えないため、依存を増やさない

### 6.6 受信パス (クライアント) のゼロコピー

```
np.empty(plan.shape, plan.dtype) で出力バッファを確保 (Python 層、§10.5)
  ↓ PyO3 で raw pointer + len を取得、GIL 解放
ScatterBuffer { ptr, len }  ← 複数スレッドが非重複領域に書くための薄い抽象
  ↓
各接続スレッド:
  loop {
      header = read_exact(32 bytes)
      if header.codec == RAW {
          let dst = scatter.slice_mut(header.offset, header.logical_len);  // 非重複を検証
          stream.read_exact(dst);      // ← カーネル → numpy 配列へ直接。中間コピーなし
      } else {
          tmp = read_exact(header.wire_len);
          decompress_into(tmp, scatter.slice_mut(...));
      }
  }
```

`ScatterBuffer` は `unsafe` を局所化するための型である。以下の不変条件を型レベルで担保する。

- 同時に発行される `slice_mut` のレンジは互いに重ならない (チャンク割り当てが重複を作らないことを、割り当て側でデバッグアサートする)
- すべてのレンジはバッファ長の範囲内である
- バッファの寿命は転送完了まで保たれる (Python 側の numpy 配列を PyO3 で保持)

これにより、受信経路のコピーは**カーネルからユーザ空間への 1 回のみ**になる。v1 は「カーネル→gRPC バッファ→bytearray→numpy 配列」で 3 回コピーしていた。

### 6.7 TCP チューニング

| 設定 | 値 | 理由 |
|------|-----|------|
| `TCP_NODELAY` | on (固定) | 小さい `FETCH` フレームが Nagle で遅延するのを防ぐ。コントロールプレーンも同じ (§5) |
| `SO_SNDBUF` | 設定可能 (既定は OS 自動) | 送信側は明示指定で 11 % 伸びた |
| `SO_RCVBUF` | OS 自動 (固定) | 明示指定はカーネルの自動調整を止め、指定値が窓の上限になる。掃引で 16 MiB 以上は自動と同値、4 MiB は 3 分の 1 に落ちた。窓を広げるのは `net.ipv4.tcp_rmem` の仕事である |
| `TCP_CONGESTION` | 設定可能 (Linux のみ) | WAN では BBR が有効な場合がある。研究上の比較対象として設定可能にする |
| `SO_REUSEADDR` | on (サーバ) | 再起動時の TIME_WAIT 回避 |

設定可能なものは設定ファイルと環境変数で上書きでき、ベンチマークで掃引できるようにする。

### 6.8 エラー処理と回復

| 事象 | 挙動 |
|------|------|
| `ERROR` フレーム受信 | ペイロード先頭の `ErrorClass` で動作が決まる (下表) |
| データ接続の切断 | その接続が担当中だったチャンクを未割り当てキューへ戻し、接続を再確立して再フェッチ。**チャンク単位で冪等なため、転送全体をやり直す必要がない** |
| 接続再確立の失敗 | 残りの接続で続行。全滅した場合のみ転送を失敗させる |
| クライアントの中断 (Ctrl-C 等) | `FETCH` の投入を停止し、投入済み分を drain (第 6.3 節) する。サーバ側の plan は TTL / LRU で回収される |
| コントロールプレーンの切断 | セッション破棄。データ接続もすべて閉じる |

**`ErrorClass` ごとの回復動作**

| クラス | 動作 | 備考 |
|--------|------|------|
| `PROTOCOL` | 接続を閉じ、転送を失敗させる。再試行しない | 実装のバグを意味する |
| `AUTH` | 転送を失敗させ、`AexConnectionError` を送出する | セッションの再確立はユーザに委ねる (下記) |
| `PLAN` | `PrepareSelection` から**自動で 1 回だけ**やり直す。2 度目は失敗させる | クライアントは `Selection` を保持しているため再発行できる |
| `REQUEST` | 転送を失敗させる。再試行しない | 要求そのものが不正 |
| `TRANSIENT` | 失敗した `FETCH` (ヘッダの `offset` / `logical_len`) をキューへ戻して再試行する | 接続切断と同じ `max_retries` (既定 3) で数える |
| `PERMANENT` | 転送を失敗させる | |

未知のクラス値を受け取った場合は `PERMANENT` として扱う (安全側)。

`AUTH` (セッションの idle timeout を含む) で自動再接続しないのは、セッションを張り直すとファイルハンドルも失われ、再 `OpenFile` まで含めた復元が必要になるためである。初版では明示的なエラーとし、Python 層での透過的な再接続は将来課題とする。なお §6.9 の通りデータ接続上の活動もセッションの `last_seen` を更新するので、転送中に idle timeout に当たることはない。

**チャンク単位の冪等性**が回復戦略の基盤である。サーバは `FETCH` を何度受け取っても同じバイト列を返せる (plan が生存している限り) ため、部分再送が自然に成立する。これは v1 の gRPC ストリームでは不可能だった (ストリームが切れたら最初からやり直し)。

### 6.9 リソース管理

| リソース | 寿命 |
|----------|------|
| セッション | `Disconnect`、コントロールプレーン切断、または idle タイムアウト |
| ファイルハンドル | `CloseFile`、またはセッション終了 |
| 転送計画 (plan) | TTL 経過、LRU 破棄、またはセッション終了 (§5.6.1)。**`FETCH` を受け取るたびに TTL を延長する** |
| データ接続 | セッション終了、または idle タイムアウト |

**plan の TTL は `FETCH` のたびに `transfer_ttl_sec` 先へ延長する**。固定期限にすると、転送所要時間が TTL を超える大きい選択 (4 GiB を WAN で、あるいはコールドキャッシュで転送する場合など) が**必ず途中で失敗する**。延長方式なら、期限切れは「クライアントが実際に沈黙した場合」にのみ起き、意図どおりの意味になる。同様に、データ接続上の活動はセッションの `last_seen` も更新する。

タイムアウト値と、接続数・同時 plan 数・セッション数の上限は第 8.2 節の設定項目で定める。「信頼できる環境」前提のため、これらは暴走防止のセーフティネットであり、公平性の保証やクォータ制御は行わない。

---

## 7. バックエンド層

### 7.1 トレイト設計

`.npy`、HDF5 (netCDF-4 を含む、第 7.5 節)、Zarr v3 (第 7.6 節) を実装済みである。v1 の `BackendFile` / `BackendDataset` / `BackendGroup` を踏襲しつつ、**論理バイト列に対する範囲読み出し**をインタフェースの中心に据える。

```rust
pub trait ArrayFile: Send + Sync {
    fn contains(&self, path: &str) -> bool;
    fn get_item(&self, path: &str) -> Result<Item>;
    fn list_children(&self, path: &str) -> Result<Vec<(String, Item)>>;

    /// パスに付いた属性。既定は空 (属性を持たない形式向け)
    fn attrs(&self, _path: &str) -> Result<Vec<(String, AttrValue)>> { Ok(Vec::new()) }
}

pub enum Item {
    Dataset(Arc<dyn ArrayDataset>),
    Group,  // 属性はパスに付くので、グループを trait object にする必要は無かった
}

pub enum AttrValue {
    Text(String),
    Array { dtype: DType, shape: Vec<u64>, data: Vec<u8> },  // little-endian, C 順
}

pub trait ArrayDataset: Send + Sync {
    fn dtype(&self) -> DType;
    fn shape(&self) -> &[u64];

    /// 選択を解決し、論理バイト列の構造を決定する。
    fn layout(&self, sel: &Selection, quality: &QualitySpec) -> Result<SelectionLayout>;

    /// 論理レンジ [offset, offset+len) を dst へ書き出す。dst.len() == len。
    /// バックエンドの内部事情 (pread か、デコードが要るか、リモートか) は
    /// 呼び出し側から見えない。
    fn read_range(&self, layout: &SelectionLayout,
                  offset: u64, len: u64, dst: &mut [u8]) -> Result<()>;
}
```

`read_range` が `&self` を取るのは、複数の接続スレッドが同一データセットを同時に読むためである。バックエンドが内部状態 (デコードキャッシュなど) を持つ場合は内部可変性で実装する (第 7.4 節)。

**呼び出し側がバッファを所有する**のが設計の要点である (第 6.5.1 節)。データプレーンは接続ごとにバッファを確保して使い回し、バックエンドはそこに書くだけでよい。これにより:

- 戻り値の enum 分岐もライフタイム管理も不要になる
- 転送中のアロケーションがゼロになる
- 新しいバックエンドを実装する際に考えることが 1 つだけになる

将来 `sendfile` によるゼロコピー経路が必要になった場合は、`fn zero_copy_source(&self, ...) -> Option<(RawFd, u64)>` のような**任意実装のメソッドを追加**し、対応バックエンドのみが `Some` を返す形にする。既存の実装を壊さずに拡張できる。

### 7.2 `.npy` バックエンド

```rust
pub struct NpyFile {
    file:         File,        // pread で読む。mmap はしない
    data_offset:  u64,         // ヘッダ直後のバイト位置
    file_len:     u64,
    dtype:        DType,
    shape:        Vec<u64>,
}
```

**ヘッダ解析は [`npyz`](https://docs.rs/npyz/) クレートに任せ、自作しない。** `npyz::NpyHeader::from_reader` は「ヘッダのみを読み、reader をデータ本体の先頭に進める」と規定されており、本設計が必要とする動作にそのまま合致する。

```rust
let mut f = File::open(path)?;
let header = npyz::NpyHeader::from_reader(&mut f)?;  // ヘッダのみ読む
let data_offset = f.stream_position()?;              // pread の基準点
// 以降は自前で pread する。npyz のデータ読み出し API (data/into_vec) は使わない。
```

npy ヘッダは「限定的な Python dict リテラル」であり、バージョンごとのヘッダ長フィールド幅の違い、64 バイト境界パディング、構造化 dtype での `descr` のリスト化など、自作すると「numpy が書けるのに読めないファイル」を生みやすい。差分テスト (第 11.3 節) で numpy を正解として照合する方針を取る以上、ここは実績あるクレートに委ねる。

**依存コスト**: `byteorder` / `num-bigint` / `py_literal` (pest) の 8 クレートが増える。ただし `npyz` の default features は空であり、`complex` / `half` feature は**データ読み出し (Deserialize) にのみ必要**でヘッダ解析には不要なため、有効化しない。tonic / prost を導入する時点で proc-macro は入るため、実質的な増加は小さい。

**読み出し**: `FileExt::read_at` (pread) を用いる。ファイル位置の共有状態がないため、複数の接続スレッドが同一の `NpyFile` を同時に読んでも安全である (`&self` のみで済む)。

**dtype の対応づけ**: `npyz::TypeStr` から `type_char()` と `num_bytes()` を取り出し、AEX の `DataType` へ写す。

| `TypeChar` | `num_bytes` | AEX `DataType` |
|-----------|-------------|----------------|
| `Bool` | 1 | `BOOL` |
| `Int` | 1 / 2 / 4 / 8 | `INT8` / `INT16` / `INT32` / `INT64` |
| `Uint` | 1 / 2 / 4 / 8 | `UINT8` / `UINT16` / `UINT32` / `UINT64` |
| `Float` | 2 / 4 / 8 | `FLOAT16` / `FLOAT32` / `FLOAT64` |
| `Complex` | 8 / 16 | `COMPLEX64` / `COMPLEX128` |

上記以外 (`TimeDelta` / `DateTime` / `ByteStr` / `UnicodeStr` / `Object` / `RawData`) と、`DType::Plain` 以外 (構造化 dtype) はエラーとする。

**制約**: いずれも npyz が返す型で確実に検出できる。

- `fortran_order: True`: `header.order() == Order::Fortran` で検出しエラー。断片計算が C 順前提のため。なお `header.strides()` が取得できるので、将来ストライドベースに一般化すれば対応は可能である
- ビッグエンディアン: `TypeStr::endianness() == Endianness::Big` で検出しエラー。対象環境が x86-64 / ARM64 であり、変換コストを払う価値がない
- 構造化 dtype / object 配列: 上表の通りエラー。`header.uses_pickled_array()` でも判定できる
- ヘッダが宣言する shape / dtype と実ファイル長の整合を開封時に検証する (壊れたファイルで範囲外 pread が起きないように)

**階層表現**: v1 互換。`/` がルートグループ、`array` が唯一のデータセット。

**動作確認済み** (npyz 0.9.1, macOS aarch64): `(1000, 200) float32` の C 順ファイルで `shape=[1000,200]`、`order=C`、`dtype=<f4 (Little, Float, 4 bytes)`、`data_offset=128` を取得し、`data_offset` を基準に自前で範囲読みした値が numpy の期待値と一致することを確認した。F 順ファイルは `order=Fortran`、`strides=[1,1000]` として正しく検出された。`BufReader` を挟んでも `stream_position()` は正しい位置を返す。

### 7.3 選択の解決

```rust
pub enum Index {
    Single(i64),
    Slice { start: Option<i64>, stop: Option<i64>, step: Option<i64> },
    Fancy(Vec<i64>),
    Ellipsis,
    NewAxis,
}

pub struct SelectionLayout {
    pub out_shape:   Vec<u64>,   // 結果の shape
    pub dtype:       DType,
    pub total_bytes: u64,
    pub kind:        LayoutKind,
}

pub enum LayoutKind {
    /// 選択結果がソース上で 1 個の連続領域 (最速パス)
    Contiguous { src_offset: u64, len: u64 },
    /// 一定間隔で並ぶ等長ブロック (arr[:, 0:50] など)
    Strided { src_offset: u64, block_len: u64, stride: u64, count: u64 },
    /// 任意の断片列 (fancy indexing、多段ストライド)
    Fragmented { /* 断片を解析的に生成するための記述 */ },
}
```

`LayoutKind` を 3 段階に分けるのは、**論理オフセットから断片へのシークを O(1) で行う**ためである。`FETCH{offset}` を受け取ったとき、`Contiguous` と `Strided` なら除算で即座に開始断片が求まる。`Fragmented` でも、選択の構造から解析的に計算できるよう記述を保つ (断片リストを実体化しない)。

これが並列ストリームの前提条件である。もし断片列を先頭から辿るしかないなら、8 本の接続がそれぞれ任意のオフセットを要求するたびに O(n) の走査が必要になり、並列化の利点が消える。

**正規化**: `Ellipsis` の展開、負のインデックス、`None` を含む `slice`、次元数不足の補完 (`arr[5]` は `arr[5, :, :]`) はすべてサーバ側の正規化ステップで解決し、`LayoutKind` を決める前に完全な形にする。numpy と同一の結果になることは差分テストで保証する。

### 7.4 デコードキャッシュ (圧縮バックエンド向け)

`.npy` には不要だが、将来の Zarr / HDF5 では必須になる仕組みなので、トレイトがこれを許す形になっていることを設計として記録しておく。

**問題**: 圧縮フォーマットのストレージチャンクは**部分デコードできない**。4 MiB だけ欲しくても、それを含むストレージチャンクを丸ごと伸長する必要がある。クライアントが論理バイト列を自由に分割する以上、複数の接続が同じストレージチャンクの別々の部分を要求することが日常的に起きる。

```
ストレージチャンク (Zarr, 10日分 = 伸長後 40 MiB)
   ┌──────────────────────────────────────┐
   │            chunk "0.0.0"             │
   └──────────────────────────────────────┘
転送チャンク (クライアントが 4 MiB で分割)
   ├────┼────┼────┼────┼────┼────┼────┼────┤
    #0   #1   #2   #3   #4   #5   #6   #7  …

#0 を接続 A、#1 を接続 B… と配ると、キャッシュがなければ
同じ 40 MiB を 10 回伸長して毎回 9 割を捨てることになる。
```

**解決**: バックエンドが伸長結果を LRU キャッシュに保持する。どの接続が要求しても伸長は 1 回で済み、転送チャンク境界をストレージ境界に揃えなくても動く。

**ただし「分割方法から独立に効率が保たれる」は測ると成り立たなかった** ([docs/benchmark-zarr.md](docs/benchmark-zarr.md))。ストレージチャンクが fetch と同じ大きさなら、キャッシュは 1 チャンクを 1 回だけ伸長し (races 0、ヒット率は理論上限の 87.5 %)、接続をまたぐ共有は起きない ── 同じ接続が自分のチャンクを 512 KiB × 8 回読むだけである。逆にチャンクが fetch より 16 倍大きいと、接続をまたいで共有する代わりに**伸長中のチャンクの競合で 1 チャンクを平均 3.5 回伸長し**、加えて大きいブロックは伸長そのものが 1 コア 764 対 1,117 MiB/s と遅い。同じデータでチャンクの切り方だけでスループットが **5.4 倍**変わる。**チャンクは fetch サイズに合わせるのが推奨**で、合わせられないときは fetch を上げる。

**キャッシュサイズの指針**: 同時に走る接続が別々のストレージチャンクを扱う場合、それらを同時に保持できなければスラッシングして伸長が繰り返される。

```
decode_cache_bytes >= 同時接続数 × 最大ストレージチャンクサイズ (伸長後)
```

この下限を割る設定は警告を出す。

**トレイトへの含意**: `ArrayDataset::read_range` は `&self` を取る (複数の接続スレッドが同一データセットを同時に読むため)。したがってキャッシュは**内部可変性**で実装する。`.npy` バックエンドはキャッシュを持たないため、この複雑さを負担しない。

**確保はしない。** チャンクは 4 MiB で、glibc が `mmap` を使い始める閾値の上にある。1 個配るたびに確保して解放すると、`mmap` / `munmap` とカーネルによるページのゼロ埋めを払い、接続の数だけアロケータのロックに並ぶ。そこで (a) チャンクは `Arc<Vec<u8>>` で持つ (`Arc<[u8]>` は `Vec` のバッファを引き取れず、変換のたびに丸ごと写すことになる)、(b) 退去したチャンクのバッファは `Arc::into_inner()` で取り戻してプールに置き、次の伸長に貸す。まだ読まれているバッファは `None` が返るので貸し出されない。バックエンドは連鎖の最後のコーデックを貸されたバッファへ書く。**この 2 つでスループットが 32.7 %、1 コアあたりが 23.9 % 上がった** ([docs/benchmark-zarr.md](docs/benchmark-zarr.md))。これは第 6.5.1 節の「転送中のアロケーションがゼロ」を圧縮バックエンドでも成り立たせることでもある。

**実装** (`backends/decode_cache.rs`): サーバ全体で 1 個の `DecodeCache` を全ファイルが共有し、キーは (データセット ID, チャンク番号) とする。ロックは単一の `Mutex` だが、保持するのは検索と挿入の間だけで、伸長とバッファへのコピーはロックの外で行う (値は `Arc<[u8]>`)。当初はシャーディングを想定していたが、シャードごとの容量がチャンクより小さくなると保持できなくなるうえ、ロック保持時間は伸長やコピーに比べて十分短いため採らなかった。同じチャンクを複数の接続が同時に取りこぼすと、それぞれが伸長する (測定で問題になれば伸長中のチャンクを待ち合わせる)。`prepare` の時点で上の下限を割っていれば、サーバは 1 度だけ警告を出す。

### 7.5 HDF5 バックエンド

`hdf5` cargo feature で有効になる (libhdf5 1.14 以降が要る)。netCDF-4 は中身が HDF5 なので同じバックエンドで開く。形式名 `hdf5` / `h5` / `he5` / `nc` / `netcdf4` と、同名の拡張子がこれに対応する。feature なしでビルドしたサーバは、これらを `REQUEST` で拒否する。

**libhdf5 はメタデータにしか使わない。** libhdf5 はスレッドセーフビルドでもプロセス全体で 1 本のロックに直列化されており、データを `H5Dread` で読むと全データ接続がこのロックに並び、第 6.4 節の並列転送が無意味になる。HDF5 2.3 で入る内部スレッドプールも「1 回の読み出しの内側」を並列化するもので、複数の接続スレッドが別々に読む本設計の形ではロックは外れない。そこで、既存の実装で性能を要するもの (hidefix、pyFAI の direct chunk read、kerchunk) と同じく、次のように分担する。

- libhdf5 (`hdf5-metno` クレート): 階層、dtype、shape、ストレージの形式、フィルタ、fill value、データのファイル内位置 (chunked ならチャンク索引) の取得。コントロールプレーンからだけ呼ぶ
- 自前: libhdf5 とは別に開いた `File` から `read_exact_at` (pread) で読み、フィルタも自前で外す。libhdf5 のロックに触れない

`.npy` と同じく、配信中にファイルが書き換えられないことを前提とする (位置はデータセットを開いたときに一度だけ取る)。SWMR には対応しない。

```rust
pub struct Hdf5File {
    file:     hdf5::File,        // メタデータ用。呼ぶたびにグローバルロックを取る
    raw:      Arc<File>,         // データ用。このファイルから開いた全データセットで共有
    cache:    Arc<DecodeCache>,  // サーバ全体で共有 (第 7.4 節)
    datasets: Mutex<HashMap<String, Arc<Hdf5Dataset>>>,  // パスごとに 1 回だけ開く
}

enum Storage {
    Contiguous(u64),        // ファイル内の開始位置
    Unallocated,            // 一度も書かれていない。全要素が fill value
    Chunked(Box<Chunked>),  // チャンク形状、フィルタ、遅延構築するチャンク索引
}
```

**ストレージの形式**:

| 形式 | 扱い |
|------|------|
| contiguous | `H5Dget_offset` の位置から pread。開封時に `offset + データ長 <= ファイル長` を検証する |
| contiguous (未割り当て) | fill value の繰り返しを返す。fill value が未定義なら 0 (libhdf5 と同じ) |
| chunked (フィルタなし) | チャンク索引から位置を引き、pread で呼び出し側のバッファへ直接読む |
| chunked (deflate / shuffle / fletcher32) | pread したチャンクのフィルタを自前で逆順に外し (チャンクごとの `filter_mask` で飛ばされたものは除く)、第 7.4 節のキャッシュに載せてから必要な範囲を写す |
| chunked (書かれていないチャンク) | fill value の繰り返しを返す |
| chunked (その他のフィルタ: szip、lzf、blosc、nbit、scale-offset、プラグイン) | 拒否 |
| compact / virtual / 外部ファイル格納 | 拒否 |

**チャンク索引**: 最初の `layout()` (コントロールプレーン) で `H5Dchunk_iter` を 1 回走らせ、C 順のチャンク番号で引ける密な表 (位置、圧縮後サイズ、`filter_mask`) を作る。一覧表示だけで全データセットの索引を作らないよう遅延させ、データプレーンが libhdf5 を待つこともないよう `read_range` より前に済ませる。各チャンクがファイル内に収まること、フィルタなしならチャンクサイズと一致することを検証する。表はチャンク数に比例するため、2^26 個を超えるデータセットは拒否する。開いたデータセットはファイルごとにパスで保持し、2 回目以降の `prepare` は索引もキャッシュ上のチャンクも再利用する。

**論理位置からチャンクへ**: `read_src(at, buf)` は要素の多次元添字からチャンク番号とチャンク内の位置を求め、同じチャンク内で連続する区間を 1 回で写す。チャンクがある軸の全長をちょうど覆う (パディングがない) 場合はその軸をまたいで区間が続くので、行ごとにキャッシュを引かずに済む。端のチャンクも HDF5 ではチャンク全体の大きさで格納されている。

**dtype の対応づけ**: 整数・浮動小数 (half を含む)・h5py の bool (enum `FALSE=0, TRUE=1`)・h5py の複素数 (compound `r`, `i`) を第 7.2 節の `DataType` へ写す。1 バイトより大きいビッグエンディアンの型、文字列、その他の compound / enum / 参照は `UnsupportedDType` とする。HDF5 2.x のネイティブ複素数型は、`hdf5-metno` が型記述に変換できないため現状は拒否される。

**階層表現**: HDF5 の階層をそのまま見せる。`list_children` は名前順で、配信できない子 (未対応 dtype、壊れたリンク、名前付き型) は一覧から除外する。1 個の文字列データセットのために兄弟全部が見えなくなるのを避けるためで、そのパスを直接 `get_item` すれば理由付きのエラーが返る。

**属性**: `attr_names` (libhdf5 の名前索引なので既に名前順) で列挙し、1 個ずつ読む。数値は **属性自身の型のまま `H5Aread`** して生バイトを取る。dtype の対応づけが大端と非数値型を既に弾いているので、出てきたバイトがそのままワイヤの形になり、変換も型ごとの分岐も要らない。文字列は可変長なら `VarLenUnicode` / `VarLenAscii` として読み、**固定長は生バイトを読んで末尾の NUL を落とす**。libhdf5 は文字集合をまたぐ変換 (固定長 ASCII → 可変長 UTF-8) を拒否し、`FixedAscii<N>` は N がコンパイル時定数だからである。netCDF-4 はテキスト属性を固定長 ASCII で書くので、これが主経路になる。

表現できない属性 (compound、enum、オブジェクト参照、opaque、文字列の配列) と、1 個で 64 KiB を超える属性は**黙って落とす**。配信できない子を一覧から外すのと同じ規約で、netCDF-4 の `DIMENSION_LIST` は参照なのでここで消える。`GetItem` と `ListChildren` のどちらも属性を付けて返す。xarray のような読み手は全変数の属性を見るので、変数ごとに 1 往復させたら意味が無い。一覧は子ごとにオブジェクトを開き直すが、メタデータのみでホットパスではない。

**external link は辿らない。** external link はホスト上の任意のファイルを指せるため、辿るとパス制限 (データルート) の外を読めてしまう。最初のファイルを開く前に `H5Lunregister(H5L_TYPE_EXTERNAL)` でプロセス全体の external link を無効にする。soft link はファイル内に閉じるので辿る。

### 7.6 Zarr バックエンド

Zarr v3 のみを読む。形式名 `zarr` と同名の拡張子がこれに対応する。cargo feature は
無く、常にビルドされる。純 Rust なので、システムライブラリを必要とする
`hdf5` / `sz` / `zfp` を optional にしている理由がここには当てはまらない。

**ストアはディレクトリである。** ノードごとに `zarr.json` が 1 個、チャンクごとに
ファイルが 1 個ある。`PathPolicy` は通常ファイルしか通さないので、ディレクトリ用の
入口 `resolve_store` を別に設ける。形式は解決の前に決める必要があるため、要求パスの
拡張子か明示された `format` で分岐し、ファイル側の経路は従来のままにする。

**ストアの外は決して開かない。** サーバはどのディレクトリを出してよいかを決め、
バックエンドはそこから出ないことを保証する。`zarr.json` もチャンクも、読む前に
同じ関門を通す。

1. 全コンポーネントが `Component::Normal` であること。`..` / 絶対パス / `.` を
   システムコール無しで弾く。アイテムパスはクライアントが与えるので、これが無いと
   `get_item("../../etc/passwd")` が実在の経路になる
2. `canonicalize` してストアルートの下にあること。外を指していたら **fill value では
   なくエラー**にする。リンク先のパスはメッセージに出さない
3. 存在しなければチャンクは fill value、メタデータは not found

チャンクごとに `canonicalize` する。open はチャンクのデコードごと (= キャッシュの
ミスごと) にしか起きないので、`realpath` 1 回は伸長の 0.1 % 未満で、キャッシュ
ヒット時はゼロである。

**メタデータ**は `serde_json` で読む。`zarr_format` は 3 のみ、`chunk_grid.name` は
`regular` のみ、`storage_transformers` は空のみ、`data_type` は 14 種のコア型の名前
(オブジェクトで書かれた拡張型は拒否) を受ける。チャンクキーは `default` と `v2` の
両方、区切りは `/` と `.` の両方に対応する。`dimension_names` は解析して捨てる
(ワイヤに載せる場所がまだ無い)。

**階層は遅延して辿る。** ストアを開くのは `zarr.json` 1 個の読み出しで済み、
`list_children` が `read_dir` を 1 回する。open 時に木を走査すると、10,000 配列の
ストアから 1 本欲しいクライアントがコントロールプレーンで 10,000 回 open することに
なる。開いた配列はパスごとに保持し、2 回目以降は索引もキャッシュ上のチャンクも
再利用する。consolidated metadata は読まない。

**属性**は `zarr.json` の `attributes` をそのまま読む。JSON は要素型を言わないので、
numpy と同じ読み方だけが正直である ── 文字列は `Text`、真偽値は `BOOL`、i64 に
収まる整数は `INT64`、それ以外の数値と混在は `FLOAT64`、一様な矩形配列はその形状の
まま。文字列の配列・オブジェクト・`null`・ぎざぎざの配列と、1 個で 64 KiB を超える
ものは**黙って落とす**。HDF5 でワイヤ形を持たない属性を落とすのと同じ規約で、
上限も同じ理由 (一覧は全子の属性を運ぶ) による。`dimension_names` は属性ではないので
ここには出ない。

**fill value** は JSON の型から作る。数値、`true` / `false`、`"NaN"` /
`"Infinity"` / `"-Infinity"`、浮動小数の生ビットの 16 進文字列、複素数の 2 要素配列。
配列の型として読めない値は**エラーにする**。誤った fill value は、書かれていない
チャンクすべてを静かに壊すからである。

**チャンク格子の走査は HDF5 と共用する** (`backends/chunks.rs`)。Zarr も HDF5 と
同じく端のチャンクをチャンク形状いっぱいにパディングして格納するので、「論理位置 →
(チャンク番号, チャンク内位置, 連続長)」の計算が一致する。sharding は同じ走査の
入れ子になるため、共用しないと同じ添字演算がリポジトリに 3 箇所できる。

**コーデック**は `bytes` (リトルエンディアンのみ)、恒等の `transpose`、`gzip`、
`zstd`、`crc32c` に対応する。`bytes` と恒等 `transpose` は格納バイトを変えないので
open 時に検査して捨て、残りをチェーンとして持つ。チェーンは適用順に書かれているので、
復号は**逆順**に回す。`gzip` は RFC 1952 なので HDF5 の deflate (zlib) とは別物で、
伸長器は共有しない。どちらも伸長後の上限をチャンク 1 個分 + 1 バイトに切って、
解凍爆弾で膨らまないようにする。`crc32c` は末尾 4 バイト (リトルエンディアン) を
検証して落とす。`blosc` は未対応で、open 時に拒否する。

**sharding** (`sharding_indexed`) に対応する。shard は複数の内部チャンクと索引を
持つ 1 ファイルで、索引は内部チャンクごとに `(offset, nbytes)` の u64 ペアが
C 順に並んだ平坦な配列である。未書き込みは `u64::MAX` の組で表す。索引は
`index_location` が `end` (既定) なら末尾、`start` なら先頭にあり、`index_codecs`
(`bytes` と `crc32c` のみ) を通っている。サイズを変えるコーデックは索引の位置を
決められなくするので拒否する。

**2 段の格子は同じ走査の入れ子呼び出しで済む。** 外側の走査 (配列を shard 形状で
割った格子) が返すのは「shard 形状の C 順配列へのバイト位置」で、それは内部格子の
走査が引数に取るものそのものである。端の shard も特別扱いが要らない — 外側が既に
配列形状で run を切り詰め、端 shard の内部格子は完全な格子で、配列の外に出る内部
チャンクは単に未書き込みになる。共用していなければ、同じ添字演算がリポジトリに
3 箇所できていた。

キャッシュのキーは `shard * 内部チャンク数 + 内部番号` で、`ChunkKey` の形は
変わらない。`decoded_chunk_bytes` は**内部チャンク**の大きさを返す。shard は
意図的に大きく、丸ごとキャッシュに載ることは無いからである。

**索引は `layout()` で作る** (HDF5 のチャンク索引と同じ規約)。データプレーンが
メタデータを待たず、壊れた索引はそれに出会ったリクエストで失敗する。shard ごとに
open 1 回と pread 1 回で、開いた配列につき 1 度だけ走る。入れ子の shard と、
隣接する内部チャンクの pread 併合は行わない。

非圧縮のチャンクもデコードキャッシュを通す。経路が 1 本になり、
`decoded_chunk_bytes` が常に値を返すのでサーバのキャッシュ不足警告もそのまま効く。

---

## 8. サーバ実装

### 8.1 構成

```rust
// コントロールプレーン: tokio + tonic
struct ControlService {
    sessions:  Arc<SessionRegistry>,
    transfers: Arc<TransferRegistry>,
    config:    Arc<ServerConfig>,
}

struct Session {
    id:      SessionId,
    token:   [u8; 16],
    files:   DashMap<u64, Arc<dyn ArrayFile>>,
    streams: AtomicU32,
    last_seen: AtomicU64,
}

struct TransferEntry {
    session:  SessionId,
    ticket:   [u8; 16],
    dataset:  Arc<dyn ArrayDataset>,
    layout:   SelectionLayout,
    expires:  Instant,
}

// データプレーン: 専用スレッド
fn data_plane_listener(registry: Arc<TransferRegistry>, cfg: Arc<ServerConfig>) {
    for stream in TcpListener::bind(addr)?.incoming() {
        std::thread::spawn(move || handle_data_connection(stream?, registry, cfg));
    }
}
```

`TransferRegistry` はコントロールプレーンとデータプレーンの唯一の共有点である。`PrepareSelection` が書き込み、データプレーンのスレッドが読む。`DashMap` による細粒度ロックとし、転送中にロックを保持しない (`Arc<TransferEntry>` を取り出してからロックを解放する)。

### 8.2 設定項目

```toml
[server]
control_addr = "0.0.0.0:50051"
data_addr    = "0.0.0.0:50052"
data_advertise_host = ""        # 空ならクライアントから見たコントロールプレーンのホストを使う

[server.limits]
max_sessions             = 64
max_fancy_indices        = 262144   # 1 選択あたりの展開後インデックス数の上限 (§5.4)
grpc_max_message_bytes   = 4194304  # gRPC の受信メッセージ上限
max_streams_per_session  = 32
max_transfers_per_session = 64
session_idle_timeout_sec = 300
data_conn_idle_timeout_sec = 300
transfer_ttl_sec         = 60

[server.transfer]
default_chunk_bytes = 4194304   # 4 MiB
max_fetch_bytes     = 16777216  # 1 回の FETCH で受け付ける上限
inline_limit_bytes  = 65536     # これ以下の選択は TransferPlan に inline 返却 (§5.6.2)。
                                # ApplyFunction の inline 上限も兼ねる
decode_cache_bytes  = 1073741824  # 伸長済みチャンクの LRU (第 7.4 節)。HDF5 の圧縮データセットが使う

[server.tcp]
sndbuf     = 0                  # 0 = OS 既定
congestion = ""                 # Linux のみ。例 "bbr"

[server.paths]
roots = ["/data"]               # この配下のみ open を許可する
```

`paths.roots` によるパス制限は「信頼できる環境」前提でも入れる。シンボリックリンク解決後の絶対パスが root 配下にあることを検証する。ディレクトリトラバーサルは事故としても起きうるため。

---

## 9. Rust クライアント

### 9.1 構成

```rust
pub struct Client {
    control: ControlClient,            // tonic (内部に tokio ランタイムを持つ)
    session: SessionInfo,
    pool:    Arc<DataPool>,
    config:  ClientConfig,
}

pub struct DataPool {
    conns:    Vec<Mutex<DataConn>>,    // 接続ごとに専用スレッドが 1 本
    queue:    Mutex<VecDeque<ChunkTask>>,  // 未割り当てチャンク (ワークスティーリング)
    rtt_est:  AtomicU64,
    bw_est:   AtomicU64,
}

impl Client {
    pub fn connect(url: &str, cfg: ClientConfig) -> Result<Self>;
    pub fn open(&self, path: &str) -> Result<FileHandle>;
    pub fn get_item(&self, h: FileHandle, name: &str) -> Result<Item>;
    pub fn list_children(&self, h: FileHandle, name: &str) -> Result<Vec<(String, Item)>>;

    /// 選択をサーバに解決させ、転送計画を得る (1 往復)。
    pub fn prepare(&self, h: FileHandle, name: &str, sel: &Selection,
                   quality: &QualitySpec) -> Result<Plan>;

    /// 解決済みの plan を dst へ流し込む。
    /// plan が失効していたら選択から一度だけ解決し直すため、選択も受け取る。
    pub fn fill(&self, plan: &Plan, h: FileHandle, name: &str, sel: &Selection,
                dst: &mut [u8]) -> Result<TransferResult>;

    /// prepare して確保済みのバッファへ fill する。Rust から使うときの近道。
    pub fn read_selection_into(
        &self, h: FileHandle, name: &str, sel: &Selection,
        quality: &QualitySpec, dst: &mut [u8],
    ) -> Result<TransferResult>;

    /// 複数の選択をまとめて解決する (RTT 隠蔽)。要素ごとに plan かエラーを返す。
    pub fn prepare_many(&self, reqs: &[SelectionRequest]) -> Result<Vec<Result<Plan>>>;

    /// 解決済みの plan 群を 1 つのチャンクキューへまとめて投入する。
    pub fn fill_many(&self, plans: &[Plan], reqs: &[SelectionRequest],
                     dsts: &mut [&mut [u8]]) -> Result<Vec<TransferResult>>;

    pub fn apply_function(&self, h: FileHandle, name: &str, sel: &Selection,
                          func: &str, kwargs: &Kwargs) -> Result<ScalarResult>;

    pub fn stats(&self) -> ClientStats;
}
```

```rust
/// 解決済みの選択。
pub struct Plan {
    pub dtype:       DType,   // サーバが返した実際の値。要求と異なりうる
    pub shape:       Vec<u64>,
    pub total_bytes: u64,
    pub is_inline:   bool,
    // request_id / ticket / inline_data は内部
}
```

**転送関数が出力バッファを引数に取る**のが設計の要点である。クライアントライブラリ側で確保して返す形にすると、Python の numpy 配列へ渡す際にコピーが 1 回入る。呼び出し側 (= Python の `np.empty`) が確保したメモリに直接書くことで、これを避ける。

**`prepare` と `fill` を分けて公開する**のは、出力バッファの shape と dtype をサーバの答えで決めるためである (§10.5)。往復回数は変わらない — `read_selection_into` も内部で `prepare` を 1 回呼ぶだけであり、分割は確保処理をその間に挟む口を開けるだけである。分けない場合、呼び出し側は選択結果の shape を自分で計算して確保することになり、サーバ側の正規化と二重実装になる。さらに、適応品質を要求すると `Plan` の dtype と shape は要求と異なりうるため、確保の前に plan を見られなければ `at(dtype=...)` を載せられない。

### 9.2 転送の流れ

```
prepare(sel):
  1. gRPC PrepareSelection → TransferPlan
     呼び出し側は plan.dtype / plan.shape / plan.total_bytes を見て出力バッファを確保する

fill(plan, dst):
  2. dst.len() == plan.total_bytes を検証
  3. plan.inline_data が非空なら (小さい選択、§5.6.2):
       dst へコピーして即 return。データプレーンを使わない
  4. ScatterBuffer::new(dst)
  5. 接続数・credit・ConnectReply.max_fetch_bytes から chunk_bytes を決め (第 6.4 節)、
     チャンクタスクを生成して queue へ
  6. 各接続スレッドを起床:
       while let Some(task) = queue.pop() {
           credit 分の FETCH を投入 (ペイロードに plan.ticket を付す)
           DATA ヘッダを読み、scatter.slice_mut(off, len) へ read_exact
           完了を記録
       }
  7. 全チャンク完了を待つ (失敗チャンクは queue へ戻して再試行)
  8. TransferResult { bytes, wire_bytes, elapsed, chunks, streams, retries, inline } を返す。
     `wire_bytes` は実際に線を流れたペイロード長で、圧縮された転送では `bytes` より小さい
```

転送の終わりにサーバへ通知することはしない (§5.6.1)。plan は TTL と LRU で回収される。

`prepare_many` / `fill_many` (gather) は step 1 を `PrepareSelections` の 1 往復に置き換え、得られた plan のうち inline でないものだけを 1 つのチャンクキューへまとめて投入する。これにより N 個の選択が N × RTT ではなく 1 往復 + 転送で完了し、かつ全接続を N 個の選択にまたがって使い切れる。

### 9.3 クライアント設定

```rust
pub struct ClientConfig {
    pub streams:        u32,     // 既定 8
    pub chunk_bytes:    u64,     // 既定 0 = サーバ推奨値に従う
    pub credit:         u32,     // 既定 16 (M6 の掃引で決定。自動計算は入れていない)
    pub connect_timeout: Duration,
    // codec を選ぶ設定項目は無い。可逆 codec が 1 つも実装されておらず、
    // 非可逆のものは転送ごとに `at(codec=...)` で選ぶ (§5.5.1)
    pub max_retries:    u32,     // チャンク再送の上限。既定 3
}
```

環境変数 (`AEX_STREAMS`, `AEX_CHUNK_BYTES`, ...) でも上書きできるようにし、ベンチマークのパラメータ掃引をコード変更なしに行えるようにする。

---

## 10. Python バインディングと API

### 10.1 バインディング方針

- PyO3 + maturin。拡張モジュール名は `aex._aex`
- abi3 (`abi3-py311` 程度) でビルドし、Python バージョンごとの wheel を避ける
- **ネットワーク I/O 中は必ず `py.allow_threads` で GIL を解放する**。これにより `get_async` が真に並行に動き、ユーザのバックグラウンドスレッドも止まらない
- `rust-numpy` で numpy 配列を扱う。Rust 側は `PyArray` から raw pointer を取り出し、`ScatterBuffer` に渡す
- Rust のエラーは Python 例外階層 (`AexError` 基底) に変換する。`ErrorClass` (§6.3) との対応は下表の通り。すべての例外は `.error_class` 属性 (str) を持ち、ユーザコードがクラスで分岐できる

  | `ErrorClass` / gRPC | 例外 |
  |---|---|
  | `PROTOCOL` | `AexProtocolError` |
  | `AUTH` | `AexConnectionError` |
  | `PLAN` / `TRANSIENT` / `PERMANENT` | `AexTransferError` |
  | `REQUEST` | `AexValueError` (`ValueError` を多重継承し、numpy 由来のコードと整合させる) |
  | gRPC `NOT_FOUND` | `AexNotFoundError` (`KeyError` を多重継承) |

### 10.2 既存 API (維持)

v1 のコードがそのまま動くことを保証する。

```python
from aex import Client
import numpy as np

with Client("localhost:50051") as client:
    f = client.open("/data/big.npy")     # FileProxy
    arr = f["array"]                     # ArrayProxy

    arr.shape, arr.dtype, arr.ndim, arr.size, len(arr)
    arr[5]                               # 単一インデックス
    arr[10:20:2]                         # スライス
    arr[[1, 5, 10]]                      # fancy indexing
    arr[:]                               # 全体
    np.asarray(arr)                      # __array__
    np.sum(arr, axis=0)                  # __array_function__ (サーバ実行)

    for name in f:                       # GroupProxy の反復 (子の名前)
        print(name, f[name])
    "array" in f                         # __contains__
    f.close()
```

`GroupProxy` は `collections.abc.Mapping` である。v1 の `__iter__` は子をプロキシで返していたが、
`__getitem__` / `__len__` / `__contains__` が揃っていながら反復だけが値を返すのは h5py・zarr と
逆で、`dict(f)` も `keys()` も使えなかった。名前を返すようにし、プロキシは `values()` /
`items()` から取る。`FileProxy` は `with` で閉じられる。

**v1 からの機能追加** (破壊的変更ではない):

```python
arr[..., 0]        # Ellipsis (v1 では失敗した)
arr[:, None]       # np.newaxis (v1 では失敗した)
arr[arr_mask]      # boolean mask (クライアント側で nonzero に展開)

f.attrs            # netCDF のグローバル属性
f["g1"].attrs      # グループの属性
arr.attrs          # 変数の属性。units は str、_FillValue は変数と同じ dtype の numpy スカラ
```

`.attrs` は `f[...]` や `for child in f` が作ったプロキシには既に載っているので往復ゼロで、`client.open()` が返す `FileProxy` だけが初回に 1 往復する。

### 10.3 追加 API (性能用)

```python
# 1. ゼロコピー受信: 呼び出し側のバッファへ直接読む
buf = np.empty((1000, 200), dtype=np.float32)
arr.read_into(buf, np.s_[0:1000])
# 繰り返し取得するループで、毎回のアロケーションと GC 圧力を避けられる

# 2. バッチ取得: 複数の選択を 1 回の往復でまとめて要求する
a, b, c = arr.gather([np.s_[0:10], np.s_[500:510], np.s_[990:1000]])
# WAN で RTT × 3 が RTT × 1 になる

# 3. 非同期取得: 転送と計算を重ねる
fut = arr.get_async(np.s_[0:10000])
do_something_else()
data = fut.result()          # concurrent.futures.Future 互換

# 4. 適応品質 (abs_error のみ実装。他は EXACT にフォールバックし警告)
view = arr.at(dtype=np.float32)        # 精度を落として転送
view = arr.at(step=(2, 2))             # 間引いて転送
view = arr.at(abs_error=1e-3)          # 誤差上限つき非可逆
view = arr.at(abs_error=1e-3, codec="zfp")  # コーデックを指名する
data = view[0:1000]
# view.applied_quality で実際に適用された品質を確認できる

# 5. 転送統計 (ベンチマーク・チューニング用)
client.stats()
# {'bytes': 1073741824, 'elapsed': 0.83, 'throughput_mibps': 1233.4,
#  'streams': 4, 'chunks': 256, 'retries': 0, 'rtt_ms': 0.08}
```

`np.s_` を受け付けるため、`read_into` / `gather` / `get_async` は `__getitem__` と同じキー形式を取る。

```python
# 6. ビュー: 選択を転送せずに保持する
view = arr.view[10:90]       # ArrayProxy。is_view が True
np.sum(view, axis=0)         # ApplyFunction に選択を載せて、サーバが 10:90 だけ読む
```

第 5.7 節の `ApplyFunction.indices` を使うための綴りである。`np.sum(arr[10:90])` と
書けないのは、`arr[10:90]` が先に評価されて転送が起きてしまい、`np.sum` が
`__array_function__` で介入する頃には ndarray になっているため。numpy には添字を
遅延させるフックがない。

ビューの shape と dtype は**クライアント側で**決める。サーバと同じ
`aex_core::selection::resolve` を `aex._aex.resolve` として呼ぶので、意味論が
食い違うことはない (プロトコル版が完全一致でなければセッションが張れないため、
別版のコードを積んだ組み合わせも成立しない)。`PrepareSelection` を使うと、
取りに行かない計画のために 1 往復と、サーバ側のプラン枠か (`inline_limit_bytes`
以下なら) 丸ごとの読み出しを消費することになる。

ビューに対してできるのは集約と添字だけで、`view` / `at` / `gather` / `read_into` は
`TypeError` にする。サーバは選択の中からさらに選択できないため、ビューへの添字は
全体を転送してからクライアントで適用する。

### 10.4 未対応 numpy 関数のフォールバック

v1 は未対応関数で**黙って全データをダウンロード**していた。`np.sqrt(arr)` が 1 GB の転送を無言で発生させる。v2 では警告する。

```python
import warnings
from aex import AexFallbackWarning

# 既定: 転送量が閾値を超えるとき警告
np.sqrt(arr)
# AexFallbackWarning: np.sqrt is not supported server-side.
#   Downloading 1024.0 MiB to compute locally.
#   Use arr[...] explicitly to silence, or set aex.set_fallback_policy(...).

# ポリシーを変更できる
aex.set_fallback_policy("warn")   # 既定。閾値超過で警告
aex.set_fallback_policy("allow")  # v1 と同じ挙動 (無言)
aex.set_fallback_policy("error")  # フォールバックを禁止 (意図しない転送を確実に防ぐ)
aex.set_fallback_threshold(64 * 1024 * 1024)   # 既定 64 MiB
```

`error` ポリシーは、本番ワークフローで「うっかり 100 GB 転送」を構造的に防ぎたい場合に使う。

### 10.5 Python 層と Rust 層の責務分担

| 責務 | 層 |
|------|-----|
| `__getitem__` のキー解析 (slice / Ellipsis / newaxis / mask 展開) | Python |
| `__array_function__` のディスパッチと警告 | Python |
| 出力配列の確保 (`np.empty`) | Python |
| 出力配列の shape / dtype の決定 | Rust (サーバ)。`Plan` を見てから確保する |
| gRPC 通信、データプレーン転送、チャンクスケジューリング | Rust |
| 選択の正規化と検証 (最終的な正しさ) | Rust (サーバ) |

キー解析を Python に置くのは、numpy のインデックス構文が複雑で、Python の型システムと密結合しているためである。ただし**正しさの最終的な保証はサーバ側の正規化**で行い、Python 側は「素直な変換」に留める。二重に実装した正規化ロジックが食い違うのを避けるため。

同じ理由で、**出力配列の shape と dtype は Python 側で計算せず、`prepare` が返した `Plan` の値をそのまま使う** (§9.1)。取得の流れは次のようになる。

```python
def __getitem__(self, key):
    indices, newaxes = parse_key(key, self.shape)          # Python 層
    plan = _aex.prepare(self._handle, self._name, indices)  # 1 往復。GIL 解放
    out = np.empty(plan.shape, dtype=plan.dtype)            # サーバの答えで確保
    _aex.fill(plan, self._handle, self._name, indices, out) # GIL 解放
    return out.reshape(with_newaxes(plan.shape, newaxes))
```

`None` (newaxis) と、単一インデックスで軸が落ちる場合の最終的な shape だけは Python 側の情報でしか決まらないため、受信後の `reshape` で与える。`reshape` はビューを返すのでコピーは発生しない。

### 10.6 メモリ管理と寿命

出力配列のメモリは Python が確保し、Python が解放する。Rust は書き込むだけで、所有権を持たない。`aex-client` の `read_selection` のように `Vec` を返す API は Rust 単体利用とテストのためのものであり、**バインディングからは使わない** (numpy 配列へ渡す際にコピーが 1 回増えるため)。

**生ポインタを渡す前の検証**は、バインディング層の責務である。`ScatterBuffer` はバッファ長と非重複しか見ないので、numpy 配列に固有の前提はここで弾く。

| 条件 | 違反時 |
|------|--------|
| C 連続 (`is_c_contiguous`) | `AexValueError` |
| 書き込み可能 | `AexValueError` |
| dtype が `plan.dtype` と一致し、リトルエンディアン | `AexValueError` |
| `nbytes == plan.total_bytes` | `AexValueError` |

`read_into` (§10.3) で条件を満たさないバッファを渡されたとき、**暗黙に一時バッファを挟んでコピーするフォールバックはしない**。ゼロコピーのための API で黙ってコピーしては目的を失うため、エラーにして呼び出し側に `np.ascontiguousarray` を書かせる。`__getitem__` の経路は自分で `np.empty` するので、これらの条件は常に満たされる。

同一バッファを指す別の numpy ビューが同時に書かれるのを防ぐため、生ポインタは `rust-numpy` の `PyReadwriteArray` 経由で取り出し、実行時の借用チェックを効かせる。

**GIL の扱い**: ネットワーク I/O は `py.allow_threads` で囲む。このクロージャには Python の型を持ち込めない (`Ungil` 制約) ため、渡すのは `&mut [u8]` だけとし、`PyReadwriteArray` のガードはクロージャの外で保持する。同期呼び出しである限り、呼び出し元のフレームとこのガードの両方が参照を持つので、GIL を離している間に配列が回収されることはない。

**転送が失敗したとき**、出力バッファの内容は不定である。`__getitem__` は例外を投げて配列を返さないので影響はないが、`read_into` はユーザのバッファなので「失敗時の内容は未定義」を docstring に明記する。

**`get_async` の寿命** (M5): 呼び出しが返った後もバッファが生き続ける必要があるため、Future オブジェクトが Rust 側で `Py<PyArray>` の強参照を保持する。Python 側で `del` されても参照数は落ちない。さらに §6.3 の drain 規則により、中断時もバッファの解放は drain 完了より後でなければならない。したがって Future の `Drop` と `cancel` は drain の完了を待つ (待っている間は GIL を離す)。

---

## 11. テスト戦略

### 11.1 Rust

| 種別 | 対象 | 方針 |
|------|------|------|
| 単体 | `aex-core` の dtype / 選択正規化 / 断片計算 / 集約 | tokio 非依存で高速に回す |
| 単体 | `aex-wire` のフレーム encode/decode | ラウンドトリップ性質テスト |
| 統合 | サーバとクライアントを同一プロセスで起動し転送 | `tests/rust/` |
| 性質テスト | ランダムな shape / 選択 / チャンク分割 | `proptest` |
| 障害注入 | 転送中の接続切断、ERROR フレーム、plan 期限切れ | 再送ロジックの検証 |

**フレームの性質テスト**が特に重要である。ワイヤフォーマットの不整合は、サーバとクライアントで定義を共有していても、エンコード/デコードの実装で入り込む。任意のヘッダ値に対する `decode(encode(h)) == h` を検証する。

### 11.2 Python

v1 の `tests/` を移植する。`test_backend_*.py` のうち npy 以外は将来のバックエンド実装時に復活させる。

### 11.3 差分テスト (最重要)

**numpy を正解として、すべての選択パターンで結果の一致を検証する**。これが AEX2 の正しさの基盤である。

```python
@pytest.mark.parametrize("key", [
    5, -1, slice(None), slice(10, 20), slice(10, 20, 2), slice(None, None, -1),
    [1, 5, 10], [-1, -2], Ellipsis, None,
    (5, slice(None)), (slice(None), 3), (Ellipsis, 0), (slice(None), None),
    (slice(2, 8), [1, 3, 5]), ...
])
def test_selection_matches_numpy(key, npy_path, aex_array):
    expected = np.load(npy_path, mmap_mode="r")[key]
    actual = aex_array[key]
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
```

集約関数についても同様に、`axis` / `keepdims` / `ddof` の組み合わせを掃引して numpy と照合する。

**非可逆の転送だけは一致では検証できない。** `ERROR_BOUND` では「全要素が要求した誤差上限以内」が検証すべき性質であり、一致ではない。Python の wheel はコーデックを無効のままビルドするので `at(abs_error=...)` は EXACT に落ち、ここの一致テストは影響を受けない。誤差上限そのものは Rust の統合テスト (`tests/rust/quality.rs`、`required-features = ["sz"]`) が、**そのビルドが持つコーデックそれぞれについて**、接続数・チャンク境界・gathered な選択のぶんだけ検証する。

さらに **v1 と v2 の出力一致テスト**を用意する。同じ `.npy` に対して v1 クライアントと v2 クライアントで同じ選択を行い、バイト単位で一致することを確認する。v1 を参照実装として使えるのは移行期の大きな利点である。

### 11.4 並列性のテスト

ゼロコピー受信は `unsafe` を含むため、以下を必ず検証する。

- **チャンク割り当てが重複を作らないこと**: 全チャンクの和がバッファ全体を過不足なく覆うことを assert
- **Miri / AddressSanitizer** で `ScatterBuffer` の単体テストを実行する
- **接続数を 1〜16 で掃引**し、すべて同じ結果になることを確認する (並列化のバグは低確率でしか出ないため、繰り返し回数を多めに取る)

### 11.5 CI

- `cargo test` / `cargo clippy -- -D warnings` / `cargo fmt --check`
- `maturin build` → `pytest tests/python/`
- `mypy python/aex/` / `ruff check`
- Linux と macOS の両方で実行する (OS 固有パスの回帰を防ぐ)

---

## 12. ベンチマーク計画

### 12.1 目的と検証する仮説

1. v1 に対する改善幅を定量化する
2. チューニングパラメータ (接続数、チャンクサイズ、credit) の最適値を決める
3. 論文・報告書に載せられる再現可能な数値を得る

以下は調査メモ (`aex/docs/throughput-research.md`) の実測値と Arrow Flight のベンチマークを根拠とした見積もり。**これは仮説であり、本章のベンチマークで検証する**。

| 環境 | v1 (Python + gRPC) | v2 (Rust + 生 TCP) の期待値 | 主たる寄与 |
|------|-------------------|---------------------------|-----------|
| localhost | 200〜800 MB/s | 3〜10 GB/s | コピー削減、GIL 排除 |
| 10 GbE | ~1 GB/s (リンク律速に近い) | ~1.2 GB/s (リンク飽和) | 改善余地は小 |
| 100 GbE | ~1.5 GB/s (CPU 律速) | 5〜15 GB/s | 並列ストリーム、コピー削減 |
| WAN (高 RTT、連続取得) | RTT 律速 | パイプライン化で RTT を隠蔽 | credit、`gather` |
| WAN (小さい選択の単発) | 1 RTT | 1 RTT (inline 返却) | §5.6.2。プレーン分離のコストを相殺 |
| WAN (小さい選択の連発) | RTT × N | `gather` で RTT × 1 | `PrepareSelections` |

**10 GbE では v1 でもほぼリンクを飽和できるため、スループットでは差が見えにくい**。v2 の価値が明確に出るのは (a) localhost / 高速リンクでの CPU 律速の解消、(b) WAN での対話的アクセスの RTT 隠蔽、の 2 つである。学内 LAN クラスタが 10 GbE の場合は、CPU 効率 (同じ帯域をより少ないコアで達成できるか) を主指標として測る。

### 12.2 測定環境

| 環境 | 用途 | 備考 |
|------|------|------|
| 手元の Mac (localhost) | 日常の回帰測定、CPU 律速の上限確認 | loopback。ネットワークの影響を排除した実装効率の指標 |
| 学内 / 研究室 LAN クラスタ | リンク飽和の確認、CPU 効率の比較 | 10 GbE 想定 |

WAN については実拠点間の確保が難しいため、必要になった時点で Linux の `tc`/`netem` による擬似 WAN (RTT・帯域・パケットロスの人工的付与) を検討する。再現性が高く、論文の評価に使いやすい。

### 12.3 測定項目

**スループット**

- データサイズ: 64 MiB / 256 MiB / 1 GiB / 4 GiB
- dtype: float32 / float64
- 選択パターン:
  - `arr[:]` — 完全連続 (最速パスの上限)
  - `arr[:, 0:K]` — ストライド (断片が多い)
  - `arr[fancy]` — ランダム行選択 (断片がランダムアクセス)
- 各 5 回測定し、中央値と四分位を報告する

**レイテンシ**

- 小さい選択 (1 行、数 KB) の往復時間
- `gather` で N 個をまとめた場合との比較 (N = 1, 10, 100)

**CPU 効率**

- 同一スループットを達成するための CPU コア数 (`/usr/bin/time` と `perf stat`)
- v1 と v2 で「1 GB/s あたりの CPU 使用率」を比較する。10 GbE でスループットが同じでも、ここに大きな差が出るはず

**パラメータ掃引**

| パラメータ | 掃引範囲 |
|-----------|---------|
| 接続数 (streams) | 1, 2, 4, 8, 16 |
| チャンクサイズ | 256 KiB, 1 MiB, 4 MiB, 16 MiB |
| credit (パイプライン深度) | 1, 2, 4, 8, 16 |
| 誤差上限 (`abs_error`) | 無損失, 1e-4, 1e-3, 1e-2, 1e-1, 1.0 |
| 回線帯域 (netem `rate`) | 1, 2, 5, 10 Gbit/s |

誤差上限を測るときは帯域も振る。圧縮は CPU とバイト数の交換なので、帯域を固定したままでは
勝ち負けが決まらない (§13 の M7)。

### 12.4 実装

v1 の `benchmarks/run_benchmarks.py` を拡張し、**v1 と v2 を同一スクリプトから同一データに対して実行**する。Python API を維持したことで、クライアントの切り替えだけで比較できる。

```python
for impl in ["aex_v1", "aex_v2"]:
    client = make_client(impl, url)
    for size, pattern, streams, chunk in sweep:
        result = measure(client, size, pattern, streams, chunk)
        results.append(result)
```

結果は CSV で出力し、プロット用スクリプトを別途用意する。**測定条件 (ホスト名、カーネルバージョン、NIC、CPU、実行時刻、設定値) を必ず記録する**。

---

## 13. 実装マイルストーン

各マイルストーンは「動作するものが手元に残る」粒度で区切る。

### M0: 足場

- Cargo workspace、`pyproject.toml` (maturin)、CI の骨格
- `aex-core`: `DType`、`npyz` によるヘッダ解析、`pread` による範囲読み出し
- テスト: npy を読んで shape/dtype が取れること

**完了条件**: `cargo test` が通り、`.npy` のメタデータを Rust から読める

### M1: コントロールプレーン

- `protos/aex.proto` の確定と `tonic-build`
- `Connect` / `OpenFile` / `GetItem` / `ListChildren` / `CloseFile` の実装
- `SessionRegistry`、`FileRegistry`、パス制限
- Rust クライアントからメタデータを取得できる

**完了条件**: Rust の統合テストでメタデータの往復ができる

### M2: データプレーン (最小)

- `aex-wire`: ハンドシェイク、フレーム encode/decode、性質テスト
- `PrepareSelection` と `SelectionLayout` (`Contiguous` のみ)、小さい選択の inline 返却 (§5.6.2)
- **単一接続**、単一チャンク、`FETCH` → `DATA` の往復
- サーバ: `pread` → 読みバッファ → `writev(header, buf)`
- クライアント: `ScatterBuffer` への `read_exact`

**完了条件**: `arr[:]` が Rust クライアントで正しく取得できる

### M3: Python バインディング

- `aex-client` に `Plan` / `prepare` / `fill` を公開 (§9.1)。出力配列を plan の shape と dtype で確保するため
- PyO3 + maturin、`aex._aex` 拡張モジュール
- Python 層: `Client` / `FileProxy` / `GroupProxy` / `ArrayProxy`
- `__getitem__` のキー解析 (Ellipsis / newaxis / mask を含む)
- `__array_function__` とフォールバック警告
- 出力バッファの検証と GIL 解放 (§10.6)
- v1 の pytest 移植、numpy 差分テスト

**完了条件**: v1 のテストスイートが v2 で通る。`benchmarks/run_benchmarks.py` が v2 で動く

**この時点で v1 との性能比較が初めて可能になる**。ここで一度測定し、以降の最適化の効果を追跡できるようにする。

### M4: 並列化と I/O 最適化

- `DataPool`: 複数接続、ワークスティーリング、credit ベースのパイプライン
- サーバ: 読みスレッドと送りスレッドの分離 (ダブルバッファリング)
- `SelectionLayout`: `Strided` と `Fragmented`、O(1) シーク
- TCP チューニング項目の設定化
- 障害注入テスト (接続断からの再送)

**完了条件**: 接続数 1〜16 で結果が一致し、スループットが接続数に対してスケールする

### M5: 性能用 API とサーバサイド計算

- `read_into` / `gather` (`PrepareSelections`) / `get_async` / `at()` (枠)
- `ApplyFunction`: 主要な集約関数、`indices` 対応
- 集約関数の numpy 差分テスト
- `client.stats()`

**完了条件**: 集約関数が numpy と一致し、`gather` で RTT が削減されることを確認できる

### M6: 評価

- ベンチマークのパラメータ掃引と結果整理
- 既定値 (streams / chunk_bytes / credit) の決定
- README、API ドキュメント、測定結果のまとめ

**完了条件**: v1 と v2 の比較データが揃い、既定値が実測に基づいて設定されている

**完了** ([docs/benchmark-m6.md](docs/benchmark-m6.md))。`streams` 4 → 8、`credit` 4 → 16、
`read_buffers` 3 → 1。VM 間の 1 GiB は v1 の 456 MiB/s に対し 10,647 MiB/s (23.3 倍)。
掃引で分かったのは、遅延のある回線では 3 つのノブが `streams × credit × chunk_bytes`
として等価に効き、遅延のない回線では接続数だけが効くということである。

### M7 適応品質 (誤差上限)

- `Codec` の上のコーデック層。アルゴリズムは外部ライブラリに委ねる (第 5.5.1 節)
- SZ3 バインディング (`sz` feature、既定 OFF)
- ブロックヘッダ、行境界での読みピースの切り出し、送受信経路
- 圧縮率・スループット・CPU の測定と、圧縮が勝ちに転じる帯域・RTT の特定

**完了条件**: `at(abs_error=...)` が誤差上限を守って転送量を減らし、どの回線条件で
勝つかが実測で示されている

**完了** ([docs/benchmark-sz.md](docs/benchmark-sz.md))。圧縮率は誤差上限 1e-4 〜 1.0 で
4.83 〜 72.7 倍。**勝ち負けを決めるのは帯域であって遅延ではない**: 圧縮側は帯域にも
RTT にもほぼ依存しない (CPU 律速) ので、無損失がその値より遅くなる回線で勝つ。
1 Gbit/s では 4.6 〜 13.8 倍。26 Gbit/s の VM 間では RTT 50 ms でも無損失が勝つ。

分岐点は**サーバに何コア使わせるか**で決まる。SZ3 は単スレッドで、圧縮は接続スレッドの
中で走るため、`streams` がそのまま圧縮の並列度になる。16 vCPU の VM で既定の 8 接続なら
分岐点は 4.7 〜 13.6 Gbit/s、コア数に合わせた 16 接続なら 7.7 〜 24.8 Gbit/s。接続数は
コア数まで線形に効き、そこで平らになる。

### M8 コーデックの選択 (ZFP)

- `Codec::Zfp` と `codec/zfp.rs` (`zfp` feature、既定 OFF)
- 転送ごとのコーデック指定 (`requested_codec` / `TransferPlan.codec` / `at(codec=...)`)
- 同一サーバ上で 2 つを交互に測る掃引

**完了条件**: `at(abs_error=..., codec=...)` が指名どおりのコーデックで転送し、
どちらをいつ選ぶかが実測で示されている

**完了** ([docs/benchmark-zfp.md](docs/benchmark-zfp.md))。枠の見積りどおり
`Codec` の値 1 つ・feature 1 つ・モジュール 1 つ・`match` の腕 2 つで載った。

**狭い回線では圧縮率が、広い回線では CPU が、それぞれ単独で効く**。論理スループットは
`min(コーデックの CPU の天井, 帯域 × 圧縮率)` でよく説明でき、1 Gbit/s の 10 点を
数 % 以内で予言する。圧縮率は SZ3 が 4.83 〜 72.7 倍に対し ZFP が 1.64 〜 5.82 倍、
速度は逆に ZFP が 1.2 〜 2.4 倍なので、**2.5 〜 7.4 Gbit/s で勝ち負けが入れ替わる**
(1 Gbit/s で SZ3 が 2.8 倍、26 Gbit/s で ZFP が 2.3 倍)。既定は SZ3 のままとした:
狭い回線ほど差が大きく、しかも狭い回線ほど圧縮を使う理由があるためである。
ZFP も単スレッドなので、`streams` が圧縮の並列度という関係は変わらない。

### M9 ベースライン (GZIP / DTYPE_CAST)

- `Codec::Gzip` (既定 ON)。比較対象としての可逆圧縮
- `Encoding::DtypeCast` (既定 ON)。float64 → float32 と float32 → float16
- 実装しないと決めたものの削除: `SUBSAMPLE`、`Codec::Lz4`、`Codec::Zstd`

**完了条件**: `at(codec="gzip")` と `at(dtype=...)` が転送でき、誤差上限付きコーデックと
同じ土俵で比べられる

**完了**。実データ (CMIP6 MRI-AGCM3-2-S、20 km、960 × 1920 の float32 3 面) で
4 MiB 行揃えブロック単位に測った結果、**汎用の可逆圧縮は生の float には効かない**:
LZ4 は 1.00 倍、ZSTD-1 が 1.31 倍、gzip-6 が 1.36 倍。効いているのはコーデックではなく
**バイトシャッフル**で、4 バイトを面ごとに転置するとどれもほぼ倍になる (shuf+zstd-1 で
2.01 倍、1.45 GiB/s)。それでも誤差上限付きコーデックには遠く及ばない: 同じデータに
SZ3 を値域の 0.1 % の誤差で掛けると 61.8 倍になる。

この測定が 3 つの判断を決めた。**LZ4 と ZSTD は入れない** (シャッフル込みでも zstd-1 の
下位互換か、両軸で gzip に勝てない)。**間引きは入れない** (stride 2 で 4 倍・誤差 2.7 % に
対し、SZ3 は 62 倍・誤差 0.1 %)。**量子化も入れない** (自前の量子化 + エントロピー符号は
同じ誤差で SZ3 に 3.5 〜 4 倍負け、誤差の質でも負ける。SZ3 が中でやっていることの
劣化版にしかならない)。

`DTYPE_CAST` だけは圧縮率以外の理由で残した。比は 2 倍止まりだが、
`wire_len == logical_len` が保たれる唯一の品質で、受信側の展開コストがゼロになり、
転送量が事前に確定し、クライアントが確保する配列も半分になる (§5.5.3)。
桁を跨ぐ場では誤差の性質も違う: 同じ最大絶対誤差で、SZ3 が小さい値を相対で中央値
17.8 % 壊すところを float16 は 0.40 % に保つ。

### M10 属性

- `Item.attrs` (`repeated Attribute`)、HDF5 の属性読み出し、`ArrayFile::attrs`
- `GroupProxy.attrs` / `ArrayProxy.attrs` / `FileProxy.attrs`

**完了条件**: データセット・グループ・netCDF のグローバル属性が Python に届き、
`_FillValue` が変数と同じ dtype で読める

**完了**。`Item::Group` を trait object にせずに済んだ ── 属性はアイテムではなく
パスに付くので、`ArrayFile` に既定実装つきのメソッドを 1 本足すだけで、他の
バックエンドは無変更のままになった。libnetcdf 4.9.3 が書いたファイルで、
グローバル属性・グループ属性・変数の属性がすべて h5py と一致することを確認した。

**次元名はまだ出ていないが、参照を復号しなくても出せる**。libnetcdf は次元ごとに
必ずデータセットを作り (座標変数が無い次元も `CLASS=DIMENSION_SCALE` と
`_Netcdf4Dimid` を持つ)、変数側は `_Netcdf4Coordinates` に次元 id を順番どおり
持つ。`DIMENSION_LIST` (オブジェクト参照) を読む必要はなく、id から同じ
`_Netcdf4Dimid` を持つデータセットを引いてその名前を取ればよい。

### M11 Zarr バックエンド

- `backends/chunks.rs` (HDF5 と共用するチャンク格子)、`backends/zarr.rs`、
  `PathPolicy::resolve_store`、`ZARR_FORMAT` の分岐
- コーデック (zstd / gzip / crc32c)、sharding、属性

**完了**。zarr-python 3.4 が既定 (zstd)、および shard 付きで書いたストアを、
zarr-python が読むのと同じ値で Python から取得できることを確認した。索引の位置は
`end` / `start` の両方で一致する。属性も `zarr.open(...).attrs` と一致する。

**測定** ([benchmark-zarr.md](docs/benchmark-zarr.md))。ファイル 1024 個と 16 個の差は
tmpfs では出ず (符号も一定しない)、コールドな virtio ディスクでは 6 通りすべてで
shard が速い (+2.7 〜 9.4 %)。転送経路は `.npy` / HDF5 と同じ形のまま、Zarr は
**接続数でより伸びる** (1 → 16 で 9.4 〜 11.2 倍、`.npy` は 6.9 倍) ── 伸長が接続
ごとの CPU 仕事だからで、これは HDF5 の gzip で見た構図と同じである。ストアを持つ
ストアを持つマシンの上の zarr-python より、ネットワーク越しの AEX2 が
3.1 〜 9.0 倍速い。**ただしそれはコーデックの速さではない** ── 1 コアあたりの
処理量は両者ほぼ同じ (472 〜 627 対 490 MiB/s) で、違うのは 1 要求で使えるコア数
(14.4 対 3.7) である。zarr-python は呼び出し側のスレッドを増やしても伸びず、
プロセスを 16 個立てて初めて追いつきかけるが、それでも AEX2 が 1.16 〜 1.63 倍速い。
**1 コアあたりでは現状わずかに負けている** (467 対 481 〜 490)。`perf` で見ると
伸長に使えているのは CPU の 6 割で、チャンクごとに 4 MiB を確保し直していることが
23 % を食っている ── デコードキャッシュが退避したバッファを持ち回れば消える
(HDF5 にも同じだけ効く)。

HDF5 と違って索引を作る必要が無いぶん素直で、**ストアの外を絶対に開かないこと**が
代わりに要になった。チャンクキーはサーバが計算するが、アイテムパスはクライアントが
与えるので、同じ関門を両方に通している。

**完了条件**: zarr-python が書いたストアを、zarr-python が読むのと同じ値で
Python から取得できる

---

## 14. 未決事項・将来課題

### 14.1 先送り項目と拡張の枠

初版では実装しないが、**後から破壊的変更なしに追加できる**ことを設計で保証する。

| 項目 | 先送りの理由 | 確保済みの枠 | 再検討時期 |
|------|-------------|-------------|-----------|
| ~~HDF5 / netCDF-4~~ | M5 の前に実装済み (第 7.5 節) | — | — |
| HDF5 の deflate 以外の圧縮 (szip、lzf、blosc など) | 需要が見えていない。libhdf5 に任せると全接続がロックに並ぶ | 第 7.5 節のフィルタの逆適用に 1 分岐足せばよい | 実データで必要になった時点 |
| ~~Zarr v3~~ | 実装済み (第 7.6 節)。`zarrs` は 0.x でストア I/O を自前に抱えており、`read_range` とデコードキャッシュの設計と噛み合わない。メタデータは JSON 1 個なので `serde_json` で自前に読む方が小さく済んだ | — | — |
| Zarr v2 / netCDF-3 | v2 の既定圧縮器は blosc なので、blosc と 1 つの仕事になる。v3 で先に転送性能を測る | `ZarrArray` を作る第 2 のパーサ (約 150 行)。dtype 文字列は `DType::from_descr` がそのまま食える | blosc に着手する時点 |
| Zarr の blosc コーデック | 評価用ストアは自分で作れる。blosc-src は cmake を要求する | コーデック連鎖の腕を 1 つ | 実データで当たった時点 |
| 入れ子の shard | 2 段で足りる。平坦なキャッシュキーが効かなくなるのもここ | `Store` を再帰にする | 2 段 shard の実データが出た時点 |
| shard 内で隣接する内部チャンクの一括 pread と fd の使い回し | 内部チャンクのデコードごとに open 1 回。**これ自体が測定対象**で、shard あり/なしの open 回数差が効くかを先に測る | `selection.rs` の span 併合を索引に当てる。接続スレッドごとに直前の shard の fd を持つ (約 10 行) | shard の掃引が要求した時点 |
| ~~可逆圧縮 (LZ4 / ZSTD)~~ | 実装しない。実データの float32 では生のバイト列に LZ4 は 1.00 倍、ZSTD も 1.07〜1.39 倍しか効かない。ベースラインとして gzip だけ入れる | — | — |
| ~~適応品質 (誤差上限)~~ | M6 の後に SZ3 で実装済み (第 5.5.1 節) | — | — |
| ~~適応品質 (キャスト)~~ | 実装済み (§5.5.3) | — | — |
| 適応品質 (値域相対の誤差) | 「選択全体の値域」を意味させる方法が要る。圧縮ブロックは自分の範囲しか見ない。`DTYPE_CAST` は形式の性質として相対精度を持つので、そちらで言える可能性がある | `QualitySpec.rel_error_bound` | 相対精度が要る実データが出てきた時点 |
| ~~間引き (SUBSAMPLE)~~ | 実装しない。`arr[::2]` が同じバイト列を既に頼めるので、増えるのは「ストライドを決めるのがどちらか」だけ。実データでは同じ誤差で SZ3 に圧縮率で桁違いに負ける | — | — |
| ~~ZFP~~ | SZ3 の後に実装済み (第 5.5.1 節)。枠の見積りどおり `Codec` の値 1 つ・feature 1 つ・モジュール 1 つ・`match` の腕 2 つで載った | — | — |
| ZFP・SZ3 以外の誤差上限付きアルゴリズム | 2 つあれば比較はできる。libpressio の Rust バインディングは未公開のままで、今も使えない | `Codec` の値 1 つと `codec::compress` / `decompress_into` の腕 1 つ | この 2 つで物足りないと分かった時点 |
| ワイヤ形を持たない属性型 (compound / enum / 参照 / 文字列の配列) と属性の書き込み | 読み出しの read-only で用途は足りる。落ちる参照は netCDF-4 の `DIMENSION_LIST` だが、次元名は `_Netcdf4Coordinates` と `_Netcdf4Dimid` から参照なしで組めるので、これが止めているものは無い | `Attribute.value` の oneof に腕を 1 つ足せばよい | 実データで compound の属性に当たった時点 |
| 変数の次元名 (`Dataset.dims`) | 属性だけでは xarray は変数の軸を名前で呼べない。クライアント側で組み立てるか、サーバが `Dataset` に載せるかを決めていない | `Dataset` にフィールドを 1 つ、または既に届いている属性からクライアントで導出 | xarray バックエンドに着手する時点 |
| TLS | 「信頼できる環境」前提。暗号化すると受信側のゼロコピーが成立しなくなる | HELLO の `flags` にネゴシエーションビットを予約 | 公開運用を検討する時点 |
| 書き込み (`DoPut` 相当) | read-only で研究目的は達成できる | フレーム種別の未使用値 (0x04、0x07 以降)。`FETCH` と対になる `PUSH` を追加可能 | 要望が出た時点 |
| ~~密な選択に対する一括 pread + 集約~~ | M4 の測定で律速と判明し実装済み (隙間 4 KiB 以下の断片を最大 1 MiB の窓で一括読み)。[docs/benchmark-m4.md](docs/benchmark-m4.md) | — | — |
| 大きい fancy 選択の効率的な送信 | `repeated int64` は 100 万要素で 8 MB になり gRPC 上限を超える。初版は `max_fancy_indices` で拒否する (§5.4) | `Index.kind` の oneof に新しい表現を追加できる (mask のビットマップ、差分 + varint、サーバ側述語評価) | 実利用で上限に当たった時点 |
| 複数サーバ分散転送 | 広域分散の本丸だが基盤が先 | `ConnectReply.endpoints` (複数返却可能) | 適応品質の後 |
| RDMA / io_uring | TCP で CPU 律速が残る場合の次の一手 | データプレーンはフレーム定義と実装が分離済み | ベンチで TCP が律速と判明した時点 |

**追加時の互換性規約**

1. クライアントは `ConnectReply.supported_codecs` / `supported_encodings` を見て要求を決める。サーバが知らない値を要求されたら `EXACT` / `RAW` にフォールバックし、`TransferPlan` で通知する
2. **フレームヘッダのサイズは変更しない**。新しい情報が必要になったら `flags` の未使用ビットか新しいフレーム種別で表現する
3. `protocol_version` はハンドシェイクで交換し、不一致は接続拒否とする

**適応品質の設計メモ**: `QualitySpec` はクライアントが明示指定する形をとり、自動判断は載せない。何を捨ててよいかはデータではなく用途が決めるので、サーバが推測できる問題ではない。

### 14.2 設計上のリスクと対応

| リスク | 対応 |
|--------|------|
| 10 GbE では v1 でもリンクを飽和でき、スループットで差が見えない | CPU 効率を主指標に (第 12.1 節) |
| `ScatterBuffer` の `unsafe` によるメモリ破壊 | 非重複の assert、Miri / ASan、接続数を掃引した繰り返しテスト (第 11.4 節) |
| 断片が細かい選択で syscall 回数が支配的になる | M4 のベンチマークで測定し、律速するなら包含領域の一括 pread を追加する (第 6.5.2 節、第 14.1 節) |
| 専用 OS スレッドモデルが多クライアント時にスケールしない | 「少数クライアント」前提を README に明記。必要なら io_uring へ差し替え (フレームプロトコルは不変) |
| `.npy` のみでは研究の主張が弱い | Zarr v3 を追加済み (第 7.6 節)。チャンク 1 個 = ファイル 1 個という別種のストレージ構造でも転送経路の結論が変わらないことを測った ([benchmark-zarr.md](docs/benchmark-zarr.md)) |

### 14.3 設計を確定する前に検証しておきたいこと

以下は実装初期に小さな実験で確かめておくと、手戻りを防げる。

1. **`pread` + `writev` で何 GB/s 出るか** — M2 の前に、`.npy` を `pread` で読んで `writev` でソケットへ流すだけの 50 行程度のプログラムを書き、localhost と LAN で測る。**ホットキャッシュとコールドキャッシュの両方**で測り、後者ではディスク律速かどうかを確認する。ここで 10 GbE を飽和できないなら、設計の前提が崩れる
2. **ダブルバッファリングの効果** — 上記を「逐次」と「読み/送り分離」の 2 通りで実装して比較する。コールドキャッシュでの差が、6.5.3 の複雑さに見合うかを判断する
3. **PyO3 で `np.empty` のバッファへ複数スレッドから書く経路が成立するか** — `ScatterBuffer` の最小実装を先に書き、Miri で検証する。Miri は PyO3 の FFI 越しには回らないため、バインディングを通した経路は接続数を掃引した繰り返しテストと ASan で確認する
4. **専用スレッドモデルのスレッド数** — 読み/送り分離により接続あたり 2 本になるため、接続数 16 × クライアント 4 × 2 = 128 スレッドで問題が出ないことを確認する

---

## 付録 A: v1 からの移行対応表

| v1 | v2 | 互換性 |
|----|----|--------|
| `aex.client.Client(url)` | `aex.Client(url)` | 互換 (`aex.client.Client` も残す) |
| `client.open(path)` | 同じ | 互換 |
| `FileProxy` / `GroupProxy` / `ArrayProxy` | 同じ | 互換 |
| `for child in f` (プロキシ) | `for name in f` (名前)。プロキシは `f.values()` | **非互換**。`GroupProxy` を `Mapping` にするため |
| `arr[key]` (int / slice / iterable) | 同じ + Ellipsis / newaxis / mask | 上位互換 |
| `np.sum(arr)` 等 27 関数 | 主要 15 関数はサーバ実行、残りはフォールバック | 結果は互換。性能特性が変わる |
| 未対応関数の無言フォールバック | 警告つき (ポリシー変更可) | `set_fallback_policy("allow")` で v1 と同一 |
| ハンドル = UUID 文字列 | `uint64` | 内部表現。ユーザからは見えない |
| `.h5` / `.nc` / `.zarr` | **未対応** | 非互換。v1 を使う必要がある |
| `fortran_order` の `.npy` | **未対応** | 非互換 |

## 付録 B: 用語

| 用語 | 定義 |
|------|------|
| コントロールプレーン | gRPC で行うメタデータ操作と転送計画の発行 |
| データプレーン | 生 TCP で行う実データ転送 |
| 論理バイト列 | 転送対象の配列を C 順で平坦化した仮想的な連続バイト列。すべてのオフセットの基準 |
| 断片 (Fragment) | 論理バイト列の一部と、ソース上の連続領域との対応 |
| チャンク | 論理バイト列を分割した転送単位。既定 4 MiB |
| 転送計画 (TransferPlan) | 選択の解決結果。dtype / shape / チャンク分割 / チケットを含む |
| チケット | 1 転送に対する capability。データプレーンで提示する |
| credit | 1 接続が完了を待たずに投入してよい `FETCH` の数。RTT 隠蔽の深度 |
| ワークスティーリング | 接続スレッドが共有キューから次のチャンクを取る動的割り当て方式 |
