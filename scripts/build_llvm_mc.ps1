# Builds a native Windows llvm-mc.exe with the AMDGPU target, for the shader disassembler, without WSL.
# The official Windows LLVM build ships no llvm-mc.exe and no AMDGPU target. This fetches only the LLVM sources the
# tool needs (sparse, shallow), builds llvm-mc alone with AMDGPU as the only target, keeps llvm-mc.exe (static CRT,
# so it runs without the VC++ redistributable) and deletes the rest. Needs git, Python 3 and Visual Studio with the
# C++ x64 tools; uses VS's CMake + Ninja, or the ones on PATH. About 2 GB of temporary space, 20-40 minutes.
#
#   powershell -ExecutionPolicy Bypass -File scripts\build_llvm_mc.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\build_llvm_mc.ps1 -Dest D:\tools\llvm-mc -NoEnv
#
# The default -Dest is outside AppData on purpose: `python` from the Python install manager (the app execution alias
# in WindowsApps) doesn't see folders that other programs create under %LOCALAPPDATA% or %APPDATA%, so an llvm-mc.exe
# there looks missing to make_game.py and the tools, although LLVM_MC names it.
param(
    [string]$Version = "21.1.8",
    [string]$Dest = (Join-Path $env:USERPROFILE ".leonida\llvm-mc"),
    [string]$Temp = (Join-Path $env:TEMP "leonida-llvm-mc"),
    [switch]$NoEnv,   # don't set LLVM_MC for the current user
    [switch]$Keep     # keep the sources and build folder (faster rebuilds)
)
$ErrorActionPreference = "Stop"

if (-not $Keep) {
    # Compare directory boundaries, including when the paths do not exist yet.
    $paths = $ExecutionContext.SessionState.Path
    $tempDir = [IO.Path]::GetFullPath($paths.GetUnresolvedProviderPathFromPSPath($Temp)).TrimEnd('\') + '\'
    $destDir = [IO.Path]::GetFullPath($paths.GetUnresolvedProviderPathFromPSPath($Dest)).TrimEnd('\') + '\'
    if ($destDir.StartsWith($tempDir, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Cleanup would delete the installation. Choose a -Dest outside -Temp or use -Keep."
    }
}

foreach ($t in "git", "python") {
    if (-not (Get-Command $t -ErrorAction SilentlyContinue)) { throw "$t wasn't found on PATH." }
}
$installer = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer"
$vs = & "$installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw "Visual Studio with the C++ x64 tools wasn't found." }
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
$jobs = [Math]::Max(1, [int]([Environment]::ProcessorCount / 2))

$cmakeDir = Join-Path $vs "Common7\IDE\CommonExtensions\Microsoft\CMake"
$cmake = Join-Path $cmakeDir "CMake\bin\cmake.exe"
$ninja = Join-Path $cmakeDir "Ninja\ninja.exe"
if (-not ((Test-Path $cmake) -and (Test-Path $ninja))) {
    $cmake = (Get-Command cmake -ErrorAction SilentlyContinue).Source
    $ninja = (Get-Command ninja -ErrorAction SilentlyContinue).Source
    if (-not ($cmake -and $ninja)) { throw "CMake and Ninja weren't found (install VS's 'C++ CMake tools' or put them on PATH)." }
}

$src = Join-Path $Temp "src"
$out = Join-Path $Temp "build"
function Remove-DirectoryRobust([string]$path) {
    if (-not (Test-Path -LiteralPath $path)) { return }
    # Strip read-only attributes on nested files and folders so Windows allows deletion
    Get-ChildItem -LiteralPath $path -Recurse -Force -ErrorAction SilentlyContinue | ForEach-Object {
        try {
            if ($_.Attributes -band [System.IO.FileAttributes]::ReadOnly) {
                $_.Attributes = $_.Attributes -bxor [System.IO.FileAttributes]::ReadOnly
            }
        } catch {}
    }
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try {
            Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction Stop
            return
        } catch {
            Start-Sleep -Milliseconds (100 * $attempt)
        }
    }
    cmd.exe /c "rmdir /s /q `"$path`"" 2>$null
}

if (-not (Test-Path (Join-Path $src "llvm\CMakeLists.txt"))) {
    if (Test-Path -LiteralPath $src) { Remove-DirectoryRobust $src }
    New-Item -ItemType Directory -Force $Temp | Out-Null
    # core.longpaths: some llvm/docs paths are about 150 characters, so with a -Temp of about 100 characters the
    # sparse checkout fails with "Filename too long" (Windows' 260-character limit). Set in the clone's config only.
    & git clone --quiet --depth 1 --branch "llvmorg-$Version" --filter=blob:none --sparse --config core.longpaths=true https://github.com/llvm/llvm-project $src
    if ($LASTEXITCODE -ne 0) { throw "git clone of llvmorg-$Version failed." }
    # libc: LLVM 23 configures only with its shared headers (llvm/CMakeLists.txt, "LLVM libc is not found").
    & git -C $src sparse-checkout set llvm cmake third-party libc
    if ($LASTEXITCODE -ne 0) { throw "git sparse-checkout failed." }
}
# A source kept by a failed LLVM 23 build of an older version of this script has no libc yet.
if ((Test-Path (Join-Path $src ".git\info\sparse-checkout")) -and -not (Test-Path (Join-Path $src "libc"))) {
    & git -C $src sparse-checkout add libc
    if ($LASTEXITCODE -ne 0) { throw "git sparse-checkout add libc failed." }
}

# -Keep (or an interrupted build) can leave sources from another -Version.
# Refuse that checkout rather than silently rebuilding the wrong release or resetting user files.
$versionError = "Cannot reuse '$src' for LLVM $Version. Choose a fresh -Temp directory for this version; existing files were kept."
if (-not (Test-Path -LiteralPath (Join-Path $src '.git'))) { throw $versionError }
$wantedCommit = & git -C $src rev-parse --verify --quiet "refs/tags/llvmorg-$Version^{commit}"
if ($LASTEXITCODE -ne 0 -or -not $wantedCommit) { throw $versionError }
$sourceCommit = & git -C $src rev-parse --verify --quiet HEAD
if ($LASTEXITCODE -ne 0 -or $sourceCommit -ne $wantedCommit) { throw $versionError }

# DIA SDK off: its headers need ATL, which the C++ build tools don't always include, and llvm-mc doesn't use it.
$body = @"
"$cmake" -S "$src\llvm" -B "$out" -G Ninja "-DCMAKE_MAKE_PROGRAM=$ninja" -DCMAKE_BUILD_TYPE=Release ^
  -DLLVM_TARGETS_TO_BUILD=AMDGPU -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded -DLLVM_ENABLE_DIA_SDK=OFF ^
  -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_DOCS=OFF ^
  -DLLVM_ENABLE_ZLIB=OFF -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_LIBXML2=OFF || exit /b 1
"$cmake" --build "$out" --target llvm-mc -j $jobs || exit /b 1
"@
New-Item -ItemType Directory -Force $out | Out-Null
# A .cmd file avoids cmd.exe's quote stripping on paths with spaces.
# Switch cmd.exe to UTF-8 before reading paths, then restore the shared console even if the build fails.
$bat = Join-Path $Temp "_build.cmd"
$batchText = @(
    '@echo off'
    'setlocal'
    'set "_leonida_cp="'
    'for /f "tokens=2 delims=:." %%c in (''chcp'') do set "_leonida_cp=%%c"'
    'chcp 65001 >nul'
    'call :build'
    'set "_leonida_exit=%errorlevel%"'
    'if defined _leonida_cp chcp %_leonida_cp% >nul'
    'exit /b %_leonida_exit%'
    ':build'
    "set `"PATH=$installer;%PATH%`""
    "call `"$vcvars`" >nul || exit /b 1"
    $body
    ''
) -join "`r`n"
[System.IO.File]::WriteAllText($bat, $batchText, [System.Text.UTF8Encoding]::new($false))

# /s with a second pair of quotes: cmd.exe strips only the outer pair, so a path holding & or ^ stays quoted.
$p = Start-Process cmd.exe -ArgumentList "/s", "/c", "`"`"$bat`"`"" -NoNewWindow -PassThru
$null = $p.Handle  # keeps the exit code readable after WaitForExit
try { $p.PriorityClass = "BelowNormal" } catch {}
$p.WaitForExit()
if ($p.ExitCode -ne 0) { throw "build failed ($($p.ExitCode)); the build folder is kept in $out" }

# Same check shader_disasm.py makes before a run: one s_endpgm on the console's GPU.
$exe = Join-Path $out "bin\llvm-mc.exe"
# Through a file: a PowerShell pipe can prepend a BOM, which llvm-mc rejects as an invalid token.
$probeFile = Join-Path $Temp "probe.txt"
[IO.File]::WriteAllText($probeFile, "0x00 0x00 0x81 0xbf`n")
$probe = & $exe --disassemble -triple=amdgcn -mcpu=gfx1013 $probeFile
if ($LASTEXITCODE -ne 0 -or -not ($probe -match "s_endpgm")) { throw "the built llvm-mc didn't disassemble an s_endpgm" }

New-Item -ItemType Directory -Force $Dest | Out-Null
# LLVM_MC is saved for future terminals, whose working directory can differ from this one.
$final = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath((Join-Path $Dest "llvm-mc.exe"))
Copy-Item -Force $exe $final
if (-not $Keep) {
    Remove-DirectoryRobust $Temp
    if (Test-Path -LiteralPath $Temp) {
        Write-Warning "Installed $final, but cleanup of '$Temp' failed."
    }
}
if (-not $NoEnv) {
    [Environment]::SetEnvironmentVariable("LLVM_MC", $final, "User")
    Write-Host "LLVM_MC set for this user (open a new terminal to pick it up)."
}
Write-Host "built $final"
