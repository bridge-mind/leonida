# Day-one report for the game executable: PE inspection, encryption check, import surface and a loader dry run.
# Reads the exe only; no game code runs.
param(
    [Parameter(Mandatory)][string]$Exe,
    [string]$GameDir
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$Exe = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Exe)
if (-not (Test-Path -LiteralPath $Exe -PathType Leaf)) { throw "$Exe not found" }
$image = Get-Item -LiteralPath $Exe
if (-not $GameDir) { $GameDir = $image.DirectoryName }
$name = [IO.Path]::GetFileNameWithoutExtension($Exe)
$out = Join-Path $root "reports\exe\$name"
New-Item -ItemType Directory -Force $out | Out-Null

$fs = [IO.File]::OpenRead($Exe); $mz = New-Object byte[] 2; [void]$fs.Read($mz, 0, 2); $fs.Close()
if ([Text.Encoding]::ASCII.GetString($mz) -ne "MZ") {
    throw "$Exe doesn't start with an MZ header. It's still packaged or encrypted, so there's nothing to inspect."
}

$size = $image.Length
$hash = (Get-FileHash -LiteralPath $Exe -Algorithm SHA256).Hash.ToLower()
"sha256 $hash  size $size" | Tee-Object -LiteralPath (Join-Path $out "hash.txt")

Write-Host "`n== PE inspection"
python -I (Join-Path $root "tools\pe_inspect.py") $Exe --game-dir $GameDir --out $out
if ($LASTEXITCODE -ne 0) { throw "pe_inspect failed" }

Write-Host "`n== GDK runtime classes and slots called through xgameruntime.lib (read from the file)"
python -I (Join-Path $root "tools\gdk_runtime_calls.py") $Exe --out $out
if ($LASTEXITCODE -ne 0) { Write-Warning "gdk_runtime_calls could not read all of $Exe; its report may be incomplete" }

Write-Host "`n== Archetype lookup candidates for mods (docs/modding.md; pass the top RVA to gta6rt --archetype-lookup)"
python -I (Join-Path $root "tools\find_hash_lookup.py") $Exe --json (Join-Path $out "hash_lookup.json")
if ($LASTEXITCODE -ne 0) { Write-Warning "find_hash_lookup could not scan $Exe" }

Write-Host "`n== What to confirm against this executable (every ASSUMPTION / UNVERIFIED / VERIFY note; called ones first)"
$calls = Join-Path $out ((Split-Path $Exe -Leaf) + ".gdk_calls.json")
$checklistArgs = @("--out", (Join-Path $out "checklist.md"))
if (Test-Path -LiteralPath $calls) { $checklistArgs += @("--gdk-calls", $calls) }
python -I (Join-Path $root "tools\exe_checklist.py") @checklistArgs
if ($LASTEXITCODE -ne 0) { Write-Warning "exe_checklist could not write the checklist" }

function Find-RuntimeLoader($build, $folders) {
    foreach ($folder in $folders) {
        $candidate = [IO.Path]::Combine($build, $folder, "gta6rt.exe")
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    }
}

$build = Join-Path $root "runtime\build"
$configs = @("RelWithDebInfo", "Release", "MinSizeRel", "Debug")
$folders = @("") + $configs
$rt = Find-RuntimeLoader $build $folders
# A gta6rt.exe older than the runtime sources (a pull since the last build) would report the old shim table, so it
# is rebuilt then too, not only when missing.
$sources = @("runtime\src", "runtime\shims", "runtime\CMakeLists.txt") | ForEach-Object { Join-Path $root $_ } |
    Where-Object { Test-Path -LiteralPath $_ }
$newest = $sources | ForEach-Object { Get-ChildItem -LiteralPath $_ -Recurse -File -ErrorAction SilentlyContinue } |
    Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
$stale = $rt -and $newest -and ($newest.LastWriteTimeUtc -gt (Get-Item -LiteralPath $rt).LastWriteTimeUtc)
if ($stale) { Write-Host "$rt is older than $($newest.Name): rebuilding it first" }
$tlsName = "gta6rt_tls_host.dll"
$hasTls = $rt -and (Test-Path -LiteralPath (Join-Path (Split-Path -LiteralPath $rt) $tlsName) -PathType Leaf)
if (-not $hasTls -or $stale) {
    if (-not $hasTls) { Write-Host "Missing runtime files: gta6rt.exe and $tlsName must be in the same folder; rebuilding the runtime" }
    $cache = Join-Path $build "CMakeCache.txt"
    if ((Test-Path -LiteralPath $cache -PathType Leaf) -and
        (Select-String -LiteralPath $cache -Pattern '^CMAKE_GENERATOR:INTERNAL=Visual Studio ' -Quiet)) {
        # The wrapper configures Ninja and cannot reuse a Visual Studio build tree.
        $commandEntry = Select-String -LiteralPath $cache -Pattern '^CMAKE_COMMAND:INTERNAL=(.+)$'
        if (-not $commandEntry) { throw "Missing CMAKE_COMMAND in $cache. Reconfigure the Visual Studio build and retry." }
        $cmake = $commandEntry.Matches.Groups[1].Value
        $folders = $configs
        $types = Select-String -LiteralPath $cache -Pattern '^CMAKE_CONFIGURATION_TYPES:[^=]+=(.+)$'
        if ($types) {
            $available = $types.Matches.Groups[1].Value -split ';'
            $folders = @($configs | Where-Object { $_ -in $available })
        }
        $existing = Find-RuntimeLoader $build $folders
        $configArgs = @()
        if ($existing) { $configArgs = @("--config", (Split-Path (Split-Path $existing) -Leaf)) }
        elseif ($types -and $folders.Count) { $configArgs = @("--config", $folders[0]) }
        & $cmake --build $build --target gta6rt @configArgs
        if ($LASTEXITCODE -ne 0) { throw "Building gta6rt in $build failed ($LASTEXITCODE)" }
    } else {
        & (Join-Path $PSScriptRoot "build_runtime.ps1")
    }
    $rt = Find-RuntimeLoader $build $folders
    if (-not $rt) { throw "runtime build did not produce gta6rt.exe in $build" }
    $tlsHost = Join-Path (Split-Path -LiteralPath $rt) $tlsName
    if (-not (Test-Path -LiteralPath $tlsHost -PathType Leaf)) { throw "runtime build did not produce $tlsHost" }
}
Write-Host "`n== Loader dry run (map, relocate, bind imports; nothing executes)"
& {
    # Keep native stderr in the report without letting Windows PowerShell stop the dry run on it.
    $ErrorActionPreference = "Continue"
    $PSNativeCommandUseErrorActionPreference = $false
    & $rt --dry-run $Exe --game-dir $GameDir 2>&1
} | Tee-Object -LiteralPath (Join-Path $out "dryrun.txt") | Select-Object -Last 40
$dryRunExitCode = $LASTEXITCODE

# Shim to-do list, grouped by DLL, most functions first.
$report = Get-Content -LiteralPath (Join-Path $out "$([IO.Path]::GetFileName($Exe)).json") -Raw | ConvertFrom-Json
$todo = $report.imports | Where-Object { $_.where -eq "shim" } | Sort-Object { -$_.functions.Count }
$lines = foreach ($t in $todo) { "{0,-40} {1,5}" -f $t.dll, $t.functions.Count }
Set-Content -LiteralPath (Join-Path $out "shim-todo.txt") -Value $lines
Write-Host "`n== Shim to-do (DLL, function count)"
$lines
$enc = $report.sections | Where-Object { $_.looks_encrypted } | ForEach-Object { $_.name }
if ($enc) { Write-Warning "High-entropy sections (still encrypted or packed): $($enc -join ', ')" }
Write-Host "`nReports: $out"
exit $dryRunExitCode
