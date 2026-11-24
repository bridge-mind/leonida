# Progress dashboard

`tools/progress_dashboard.py` turns outcomes the project already produces into one page that answers three questions:
what works, what changed since it last passed, and what blocks next. It runs locally, reads only, and writes nothing
into the repository.

```powershell
python -I tools/progress_dashboard.py --out ..\leonida_progress --run-tests --github
start ..\leonida_progress\progress.html
```

`--out` must be outside the repository; the tool refuses a folder inside it, so its output can't be committed by accident.
The page is one self-contained HTML file: no network, no packages, it opens from disk. The project logo
(`assets/logo.png`) is embedded in it, and the page follows the system theme until you pick one with its theme button.

## Sources

Each source is optional. A missing one is listed under "Sources" on the page; it is never counted as zero or as a failure.
For `--junit`, keep the original XML encoding. UTF-16 reports and reports with a supported encoding declaration
are read directly; there is no need to re-save them as UTF-8.

| Panel | Source | Unit |
|---|---|---|
| Unit tests | `--run-tests` runs `tests/test_*.py` with the selection `ci.yml` uses (standard-library files with a `__main__` block), one process each; the outcome and the commit it ran on are kept in `<out>/unit_results.json`. That is CI's standard-library loop only: `tests/semantic` (the translator suite) and files that need pytest or numpy are not run, so the panel is not the whole test suite; feed those through `--junit` (`python -m pytest tests/semantic --junitxml <file>`) to see them in the runtime-tests panel | test file |
| Runtime tests | `runtime/build/Testing/Temporary/LastTest.log` and any `--junit` file (CTest `--output-junit` or pytest `--junitxml`). CTest logs and XML reports are age-checked by file modification time | test |
| GPU tests | `--gpu-log <file>[=<GPU, driver, OS>]`: the saved output of `runtime/test/shaders/check.py`, one group per GPU; a log older than the last commit touching `runtime/test/shaders` or `tools/rdna2spirv.py` counts as needing a re-run | test program |
| Translator contract | the contract table in [design.md](design.md#translator-status-and-runtime-contract), each row classified by how it says it was confirmed | contract row |
| Runtime imports | the measured tables in [runtime_imports.md](runtime_imports.md) | import |
| Shader corpus | `--work <work>`: counts per stage from `<work>/shaders/dxbc/index.json` and `<work>/shaders/spirv/index.json` | shaders, aggregated per stage |
| Pull requests into main | `--github`: every pull request into `main` of the main repository (`--repo`, default `ItsNotRexed/Leonida`), grouped by the item ID its title names (`D3D-86: …`, `… (SHADER-339)`) | pull request |
| D3D12 layer | `--github`: the `D3D-` table of [#313](https://github.com/ItsNotRexed/Leonida/issues/313), read through `gh` and checked against the merged and open pull requests | item |
| CI on main | `--github`: the latest completed run of each workflow on `main` | workflow |

Every run with `--github` reads the pull requests again, so a merge into `main` shows up on the next run as
IMPLEMENTED, even before a tracking table (such as #313's PR column) is edited. A pull request closed without merging is
left out. The page also lists the 30 latest merges, and the trend chart follows the share of pull requests merged and
of item IDs with a merged pull request. `gh pr list` reads at most 5000 per state; the Sources table says when that
limit is reached.

With `--github --login <account>` the page also lists that account's claims in the coordination issue (and the issue
it continues) and its pull requests. Nothing is posted: every `gh` call is a read.

## How a status is decided

| Status | Unit tests and CTest | Contract row | Import | Pull request | D3D- item |
|---|---|---|---|---|---|
| VERIFIED | passed, and nothing it depends on changed since | names a GPU check and leaves nothing open | image with 0 unimplemented imports | — | — |
| IMPLEMENTED | — | confirmed by an LLVM/Mesa reference, a corpus count or a CPU test, not a GPU | bound to a shim, the host or a module | merged into `main` | a PR merged into `main` and none open |
| PARTIAL | passed before a change to the test, `make_game.py`, or a Python file under `tools/`, `scripts/` or `runtime/test/shaders/` (CTest: before the last commit touching `runtime/`) | waits for a GPU measurement, or names a GPU check with an item still open | — | open | pull request open |
| FAILED | failed on its last run | — | — | — | — |
| MISSING | — | — | unimplemented | — | unclaimed |
| UNKNOWN | skipped, or git can't tell what changed | states no confirmation method | — | — | claimed, no code yet |

The contract classification reads the row's own words; the matched words are shown in each cell's details, so a
wrong class is visible and the rule list in the tool (`CONFIRMATION`) can be corrected.

Each metric is a ratio with its numerator, denominator and formula on the page. There is no overall percentage: the
panels count different things. The trend chart breaks a line where a metric's formula changed between snapshots.

## No game data

The page and the JSON carry counts, pass/fail, error classes and the project's own document text. The corpus panel
stores per-stage numbers only: no shader name, carved file name or path into the game copy. One-line failure details
are kept with each absolute path replaced: a path inside the repository becomes its relative path, any other path
(work folder, home folder, temp folder) becomes `<path>`, file name included, since a file name outside the
repository can be a carved shader's name.
