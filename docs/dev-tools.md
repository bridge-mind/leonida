# Development tools

From the Leonida checkout root, use Python 3 to list the development tools:

```powershell
python -I scripts/dev.py --help
python -I scripts/dev.py discover
python -I scripts/dev.py discover --json
```

From another directory, pass the path to `scripts/dev.py`, quoting it if needed.
The inventory always describes the checkout containing that script.

`boot-trace` points to the analyzer for startup JSONL and D3D12 stderr logs.
`guest-scene` points to the headless synthetic guest runner. Run a listed help
command from the checkout root to see the producer's options. The text
inventory displays commands for PowerShell on Windows and POSIX shells
on other platforms. The inventory
checks for each producer script and lists its prerequisites without checking
those prerequisites or running the producer. Live attach, pause, step, input,
and screenshots are unavailable through this entry point.

## JSON output

`discover --json` writes one object to stdout:

- `schema_version` identifies the format. The current version is `1`.
- `tools` lists entries by `id`. Each `path` uses forward slashes and is
  relative to the checkout root. `present` indicates whether the script is
  a file. `help_argv` is a list of arguments for a command run from the checkout
  root. Its first argument is the Python executable running discovery. It
  is `null` when the script is missing. `requirements` lists prerequisites
  for using the producer.
- `live_controls` contains an `available` boolean and a `reason`.

Exit code `0` indicates that the inventory was emitted, including when a
producer script is missing. Invalid CLI arguments exit with code `2`.

## Analyze a boot trace

Use the existing analyzer through the development CLI:

```powershell
python -I scripts/dev.py boot-trace --help
python -I scripts/dev.py boot-trace "logs/startup trace.jsonl"
python -I scripts/dev.py boot-trace "logs/startup trace.jsonl" --d3d12 "logs/d3d12 stderr.log" --json
```

Pass a startup JSONL from `gta6rt --trace`, a D3D12 stderr log from
`gta6rt --trace-d3d12`, or both. Relative input paths are interpreted from
the directory where you run the CLI. From outside the checkout, give the
path to `scripts/dev.py` and keep the trace paths relative to your directory.

The analyzer supplies the help, output, diagnostics and exit status. If one
input fails but the other can be read, `--json` can still emit a partial
summary on stdout with a diagnostic on stderr and exit code `2`. If neither
input can be analyzed, there is no JSON summary. The summary describes the
recorded events, and the exit status reports the analysis result.

You can also run `python -I tools/boot_trace.py` with the same arguments.

## Run a synthetic guest scene

Use the existing headless runner with a loader and guest built on Windows
with Vulkan, static SDL3 and `LEONIDA_GUEST_SCENE=ON`. Supply the synthetic
M1.5 shader fixture root containing `spirv/`. See the
[runtime instructions](../runtime/README.md#a-600-frame-headless-guest-scene)
for the build, fixture and Wine setup.

```powershell
python -I scripts/dev.py guest-scene --help
python -I scripts/dev.py guest-scene --loader "build/gta6rt.exe" --guest "build/tiny_guest_draw.exe" --shaders "m15 shaders" --out "scene run" --negative-controls
```

Pass `--out PATH` or `--out=PATH` with a new or empty directory. Relative
paths are interpreted from the directory where you run the CLI, including
when you invoke `scripts/dev.py` by its full path from another directory.

The CLI forwards the runner's exit code and diagnostics. Exit code `0`
means the scene checks passed. Exit code `1` can indicate a scene check
failure or a failure before the runner can write a report, such as an
output directory it cannot create. Exit code `2` indicates a usage or
prerequisite error.

When the runner writes `report.json`, it prints the path. Open that file
for the scene result and per-run details. For example, if the full run
starts and the loader exits with an error, `runs.full.returncode` records
the loader's exit code while the CLI returns the runner's exit code.
The output directory also holds any logs, frame records and PPM images
the runner produced. If the runner rejects an occupied directory, any
report already there belongs to an earlier run.

You can also invoke `python -I scripts/run_guest_scene.py` directly.
