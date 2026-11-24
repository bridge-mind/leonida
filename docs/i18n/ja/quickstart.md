# Leonida クイックスタート（`ja`）

> [!NOTE]
> 英語のドキュメント（[`README.md`](../../../README.md)、[`CONTRIBUTING.md`](../../../CONTRIBUTING.md)、
> [`docs/README.md`](../../README.md)）が**正式版**です。このページは安定したセットアップとビルド手順のみを
> まとめたもので、commit `003e781` に同期しています。日々の進捗は [`README.md`](../../../README.md)
> を参照してください。

## Leonida とは

Leonida は、Xbox Series 版 Grand Theft Auto VI を Windows 上でネイティブに動かすための**静的リコンパイル**
プロジェクトです。Xbox Series の CPU は PC と同じ x86-64 なので、ゲームのコードはそのまま読み込まれて
ネイティブに実行され、本体側のシステムソフトウェアは shim として再実装されます。本体用にコンパイル済みの
RDNA2 GPU プログラムは Vulkan 向けの SPIR-V に翻訳され、モデル・テクスチャ・音声はあなたのパッケージから
直接読み込まれます。これらはすべて**あなたの PC 上で、あなた自身のゲームのコピーから**行われます。

> [!IMPORTANT]
> このリポジトリにはゲーム、キー、実行ファイルは**含まれません**。入手方法も説明しません。自分用のコピーが
> 必要で、そのファイルはあらかじめディスク上で読める状態になっている必要があります。ここには復号や保護の
> 回避を行うものは何もありません。ビルドが生成するもの（翻訳されたシェーダー、抽出されたアセット、
> runtime フォルダー、セーブデータ）はすべてあなたのゲームに由来し、あなたの PC に残ります。これらの
> ファイルを共有しないでください（[`CONTRIBUTING.md`](../../../CONTRIBUTING.md)）。

プロジェクトはまだ初期段階で、**ゲームはまだ起動しません**。

## 前提条件

- **自分用の Xbox Series パッケージ**（バージョン `1.0.2.1`）。`MicrosoftGame.Config` と `.rpf` アーカイブが
  プレーンなファイルとして置かれているフォルダーが、ゲームフォルダーです。
- **Windows 10/11 x64** と、作業ファイル用に約 **400 GB** の空き容量（`--work` で別ドライブに置けます）。
- [Python 3.12+](https://www.python.org/downloads/)（インストール時に "Add python.exe to PATH" を有効に）。
- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0)。
- [Vulkan SDK](https://vulkan.lunarg.com/)（`spirv-as` / `spirv-val` 用）。
- **AMDGPU target** を含む [LLVM 21+](https://github.com/llvm/llvm-project/releases)（`llvm-mc` 用）。
  - Windows 版の公式 LLVM には AMDGPU target が**含まれていません**。
  - Windows では、リポジトリのルートで次のコマンドを実行すると、AMDGPU target 付きのネイティブ `llvm-mc.exe` を
    ビルドし（WSL 不要。`git` と下記の Visual Studio の C++ ツールが必要）、`LLVM_MC` を設定します:

    ```powershell
    powershell -ExecutionPolicy Bypass -File scripts\build_llvm_mc.ps1
    ```
  - または WSL に `llvm-21` を入れて、`LLVM_MC` にコマンド全体を設定します:
    `LLVM_MC="wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc"`
    （[docs/design.md](../../design.md#disassembling-shaders)）。
  - Linux では `PATH` に `llvm-mc` があれば十分です。
- **Visual Studio 2022**（**C++ によるデスクトップ開発**）と **Windows SDK 10.0.26100 以降**。
  `runtime` ステージで必要です（バージョン 1.2 の root signature を使い、古い SDK には宣言がないためです）。
- [ffmpeg](https://ffmpeg.org/download.html) を `PATH` に（任意: XMA2 ストリームのデコード。`--wav` では必須）。

## ビルド方法

ポートはビルド済みで配布されているものではありません。**あなたの PC 上で、あなた自身のコピーから**ビルドします。

### 簡単な方法（Windows）

1. 上記の前提条件をインストールします。
2. サブモジュール込みでリポジトリを取得します。

   ```powershell
   git clone --recursive https://github.com/ItsNotRexed/Leonida
   ```

   GitHub の "Download ZIP" では `third_party/rpf9-explorer` が含まれません。`git` があればビルドが自動で
   取得し、なければ必要な操作を表示します。
3. **`Build.bat`** をダブルクリックします。

すると、ゲームのコピーが自動で見つかります。ダウンロード、デスクトップ、ドキュメント（OneDrive など
Windows が保存する場所を含む）、Games、ホームフォルダー、各ドライブの先頭から 2 階層までを探し、いくつかの
ファイルサイズからバージョンを判定します（大きなファイルは読みません）。複数ある場合はどれを使うか確認し、
見つからない場合はフォルダーの貼り付けを求めます。開始前に上記の前提条件と空き容量を確認し、足りないものを
まとめて表示します。出力先は `Documents\Leonida` です。中断したビルドは続きから再開します。

### コマンドライン（`make_game.py`）

```powershell
python -I make_game.py --find                          # この PC で見つかったコピーを一覧
python -I make_game.py --pick                          # フォルダーダイアログでゲームフォルダーを選ぶ
python -I make_game.py --check                         # コピーを確認するだけでビルドしない
python -I make_game.py --check --full-check            # すべてのファイルの SHA-256 も比較（低速）
python -I make_game.py --game "D:\path\to\your\package" --out "D:\Games\Leonida" --work "E:\leonida-work"
python -I make_game.py --stages shaders,assets         # 一部のステージだけ
python -I make_game.py --game "D:\path\to\your\package" --wav   # Opus を PCM WAV にもデコード（大容量）
python -I make_game.py --wave64-emulate off            # auto（既定）、on、off のいずれか
```

ステージは `check` → `shaders` → `assets` → `runtime` → `pipelines` の順に実行されます。`pipelines` は
翻訳済みの各シェーダーをあなたの GPU ドライバー上で生成してパイプラインキャッシュを保存するので、初回起動時の
コンパイルが不要になります。最も時間がかかるのはシェーダーの翻訳で、`--jobs` 個のプロセスで実行されます
（既定では CPU スレッド数の半分）。

- **`--wave64-emulate`**: wave64 の compute / mesh シェーダーを、どのサブグループサイズでも正確になるように
  翻訳するかどうかを決めます。NVIDIA と Intel の GPU は 32 レーンのサブグループなので必要です。既定の `auto` は
  `vulkaninfo`（Vulkan SDK に含まれる）に問い合わせ、すべての GPU が 64 レーンのサブグループを報告しない場合、
  または判定できない場合に有効にします。ただし runtime が使う GPU が、AMD の Windows ドライバーで wave64 を
  ネイティブに実行できる RDNA2 のカードや iGPU の場合は、そのドライバーのシェーダーコンパイラーが
  エミュレーションされた LDS コードでクラッシュするため（#857）、オフのままにします。設定は翻訳済みシェーダーの
  隣に保存され、変更すると翻訳し直しになります。
- **`--work`**: 中間ファイル用のフォルダーです。最低でも 400 GB 空いているドライブを指定してください。
- **`--wav`**: 既定では無効です。16-bit PCM WAV は Ogg Opus より数倍大きいためです。
- **Linux と Wine/Proton**: `make_game.py` は Linux でも動作します。ローダーは Windows プログラムなので
  Wine または Proton で動きます。`runtime` ステージは Linux 上で clang-cl と lld-link（LLVM 19 以降）を使い、
  [xwin](https://github.com/Jake-Shadle/xwin) が `XWIN_DIR` または `~/.local/share/xwin-sdk` に用意した
  MSVC CRT と Windows SDK に対して `runtime/build/windows` へクロスビルドします
  （[docs/testing.md](../../testing.md#the-windows-runtime-on-linux-under-wine-or-proton) に一度だけ実行する
  `xwin` コマンドがあります）。これらがない場合、このステージは既定のステージから外れ、明示的に指定すると
  何が足りないかを作業開始前に表示します。別の場所で作った Windows 版も使えます:

  ```sh
  python3 -I make_game.py --runtime-build ~/Downloads/runtime-windows
  ```

  どちらの場合も `gta6rt.exe` と `gta6rt_tls_host.dll` を `--out` にコピーし、隣に `gta6rt.sh` を書き出します。
  これは `gta6rt.exe` を `wine`、または `WINE` で指定した Wine（Proton なら `files/bin/wine`）で起動します。
  Windows で同じオプションを指定すると、ビルドは省略されます。

## よくある問題

- **GitHub の "Download ZIP" を使った**: ZIP には `third_party/rpf9-explorer` サブモジュールが含まれません。
  `git clone --recursive` を使うか、リポジトリのフォルダーで `git submodule update --init --recursive` を
  実行してください。
- **`llvm-mc` が `gfx1013` や AMDGPU を拒否する**: Windows 版の公式 LLVM を使っている可能性が高いです。
  リポジトリのルートで `powershell -ExecutionPolicy Bypass -File scripts\build_llvm_mc.ps1` を実行するか、
  `LLVM_MC` を WSL の `llvm-mc`（`llvm-21`）に向けてください。
- **CMake が "needs the Windows SDK 10.0.26100 or newer" で止まる**、または MSVC が `Desc_1_2` は
  `D3D12_VERSIONED_ROOT_SIGNATURE_DESC` のメンバーではないと報告する: Windows SDK を `10.0.26100` 以降に
  更新し（Visual Studio Installer、または `winget install Microsoft.WindowsSDK.10.0.26100`）、
  `runtime/build` を削除してから構成し直してください。
- **`C:` ドライブの空きが足りない**: `--work "D:\leonida-work"` を指定して中間ファイルを別のドライブに置きます。
- **Windows がビルド直後のテストプログラムの実行を拒否する**（"An Application Control policy has blocked this
  file"）: Windows 11 のスマート アプリ コントロールです。
  [docs/testing.md](../../testing.md#known-problems) を参照してください。

## コントリビュートと GPU テスト

- GPU テストの手順（ゲームファイル不要）: [docs/testing.md](../../testing.md)。結果は
  [#166](https://github.com/ItsNotRexed/Leonida/issues/166) に、GPU とドライバーごとに 1 コメントで投稿します。
- コントリビュートのルール、ゲームデータを持ち込まない方針、作業の調整:
  [`CONTRIBUTING.md`](../../../CONTRIBUTING.md)。
