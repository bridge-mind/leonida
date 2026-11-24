# Guide for coding agents

Read this before changing anything here. [CONTRIBUTING.md](CONTRIBUTING.md) has the rules for every contributor and
[docs/design.md](docs/design.md) the formats and the translator's runtime contract; this file adds what an agent
tends to get wrong.

## What this is

A port template for the Xbox Series build of GTA VI, built on the player's PC from their own copy:
`make_game.py` carves and translates the shaders (RDNA2 → SPIR-V), exports the assets and builds `runtime/`, a PE
loader that maps the game's executable and resolves its imports to shims. Nothing of the game is in the repository.

## Where things are

- The notes under `docs/` record measured formats, indexed by [docs/README.md](docs/README.md); `docs/design.md` is the one quoted most.
- `scripts/` holds the GPU/Wine test runners (`run_gpu_tests.py`, `run_wine_tests.py`), the translator suite runner
  `run_semantic_tests.py`, `check_no_game_data.py` (see the first hard rule below) and PowerShell build helpers.
- `viewer/` is the map viewer: `index.html` plus `serve.py`, a local-only server for the exported world.
- `third_party/` holds the `rpf9-explorer` submodule (`tools/rpf9dump` needs `git submodule update --init`), the vendored `xtaskqueue` the `xgameruntime` shim links and the vendored `bcpack-rs` rANS decoder of the BCPack calibration tools.
- `game/xbox-1.0.2.1.json` is the per-file size and SHA-256 manifest of the Xbox package — hashes only, no game content.
- `tests/semantic/` is the translator regression suite; a new translator rule comes with a small hand-written `.asm` fixture there.

## Hard rules

- **No game data, ever**: no archives, executables, carved files, shaders in any form (DXBC, FSSM, disassembly,
  SPIR-V), textures, audio, or logs and reports that quote them, in commits, issues, pull requests or test
  fixtures. A test that needs a shader builds its input by hand or reads it from the user's work folder.
  `python -I scripts/check_no_game_data.py <path> ...` reads the first bytes of the files you name, so it catches a
  renamed file the extension check in CI misses.
- **No keys, decryption or protection workarounds.** If a format turns out to be encrypted, write that down in
  `docs/design.md` and stop there.
- **Nothing online**: no Xbox Live or Rockstar services or domains.
- **Fix, don't disable**: a shader the translator rejects gets a rule, not a skip list. A new rule goes into the
  contract table in `docs/design.md` with how it was confirmed (a GPU test, a Mesa/LLVM reference, or a count over
  the corpus).

## Claim before you start

- Work is claimed in the coordination issue before a pull request is opened: the open issue titled "Coordinating
  who works on what". It moves to a new issue when the thread gets long, so look it up instead of using an old
  number:

  ```sh
  gh issue list -R ItsNotRexed/Leonida --state open --search "Coordinating who works on what in:title"
  ```

  With no `gh` auth, the search link in [CONTRIBUTING.md](CONTRIBUTING.md) works in a browser; say so when you cannot
  read the claims at all rather than guessing that a thing is unclaimed.

  Check its tables and the open pull requests first. If someone has claimed the item or has a pull request for it,
  stop and tell the user instead of writing it a second time.
- Put the item ID from that issue in the pull request, and `Depends on: #N` for every pull request yours builds on.
- Found work you won't do yourself (a bug, a missing method, a stale doc, a follow-up of your own pull request)?
  Post it in the coordination issue as `finding: <one line>`, followed by where it is (file and line), why it matters,
  and whether it needs game files or particular hardware. The coordinator checks it and lists it for others.
- For the D3D12 → Vulkan layer (`runtime/shims/d3d12_*`, `runtime/src/vulkan_*`), also read
  [#313](https://github.com/ItsNotRexed/Leonida/issues/313): most of that work builds on a few open pull requests,
  and it lists which D3D12 methods are still open.

## Before you guess, look

- The format or behaviour may already be measured in `docs/design.md`. Quote the file and offset a claim comes from,
  or call it a hypothesis.
- Count over the whole corpus, not a sample: a rule that holds for 400 shaders can fail on the other 26 000.

## Test without the game first

None of these need game files. Most run in seconds - the translator suite takes a few minutes (433 files and 5 821
tests in 224 s on 16 cores) and the two builds longer - so run one translator test file while you iterate and the
whole suite once before the pull request:

```powershell
python -m compileall -q make_game.py tools tests
python -I make_game.py --help
python -I scripts/run_semantic_tests.py tests/semantic   # translator suite, one file per core; no LLVM, SPIR-V tools or GPU needed
python -m unittest discover -s tests/semantic -p test_<name>.py   # one translator test file
python -I tests/test_<name>.py                     # standalone standard-library unit test
node --test tests/test_viewer_streaming.cjs        # viewer streaming; needs Node 22+
dotnet build tools/rpf9dump/rpf9dump.csproj -c Release
cmake -S runtime -B runtime/build; cmake --build runtime/build --config RelWithDebInfo
ctest --test-dir runtime/build -C RelWithDebInfo --output-on-failure
```

`gta6rt` and the shims are Windows-only, so Linux and macOS configure the portable subset of the runtime tests. The
runtime tests that need the retail package read `LEONIDA_GAME_DIR` (and `LEONIDA_MOVIE`) and print `SKIP` when
unset; `loader_test` writes small PE images to `%TEMP%\leonida_loader_test`, which antivirus can block.

For pytest-based tests (the test files with `import pytest`, or no `__main__` block: they fail to import under
`python -I`), use a virtual environment and install the pinned dependencies in
[`tests/requirements.txt`](tests/requirements.txt). From the repository root, run `python -m pytest` with the
test file you want to check.

```powershell
python -m pip install -r tests/requirements.txt
python -m pytest tests/test_user_sgpr_census.py
```

With a work folder already built, a translator change is checked on a slice before the full run, and the result is
compared with the run before it:

```powershell
python -I tools/rdna2spirv.py <work>/shaders/dxbc/disasm --out <scratch> --limit 200
python -I tools/rdna2spirv.py <work>/shaders/dxbc/disasm --out <scratch> --shard 0/8   # one of 8 parallel slices
```

Shard runs write `index.part*.json`; merge them into `index.json` with `--merge`. Report translated and
`spirv-val`-valid counts before and after; a change that lowers either is a regression.

## Conventions

- Python tools run with `python -I` and use only the standard library unless a file says otherwise; the exceptions
  are the texture and model tools that import numpy or Pillow: `bc_feature.py`, `amd_swizzle.py`, `tile_probe.py`,
  `swizzle_search.py`, `tex_export.py`, `tex_mode_probe.py`, `calibrated_texture.py`, `world_grid_atlas.py`,
  `tex_headers.py` and `skinned_glb.py` (which fall back to the standard library when numpy is absent).
- A text file a tool writes names its line ending - `open(path, "w", encoding="utf-8", newline="\n")` - so the same
  input gives the same bytes on every platform; `tests/test_text_write_line_endings.py` catches a writer that forgets.
- Keep `make_game.py` working on Windows first: paths through `os.path`, no bash in the build.
- In the pull request, say that it was AI-assisted, what you tested and on which GPU and driver.
