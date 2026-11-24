# Builds runtime\build\gta6rt.exe with MSVC at BelowNormal priority, on half the cores by default.
# Uses VS's CMake + Ninja when the "C++ CMake tools" component is installed (or native Windows cmake + ninja on PATH),
# otherwise calls cl.exe directly.
# The GameInput controller shim (SDL3, docs/gameinput.md) is built only on the CMake path: with an installed SDL3, or
# with -FetchSdl3, which lets CMake download the pinned SDL3 release. The cl.exe fallback never builds it.
param(
    [switch]$Debug,
    [switch]$FetchSdl3,
    [ValidateRange(1, 2147483647)][int]$jobs = [Math]::Max(1, [int]([Environment]::ProcessorCount / 2))
)
$ErrorActionPreference = "Stop"

function Invoke-ToolProbe([string]$Exe, [string[]]$ProbeArgs) {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $global:LASTEXITCODE = 0
        $text = (& $Exe @ProbeArgs 2>$null) -join "`n"
        if ($LASTEXITCODE -eq 0) { return $text }
    } catch {
    } finally {
        $ErrorActionPreference = $prev
    }
    return $null
}

function Find-BuildTool([string]$Preferred, [string]$CommandName, [scriptblock]$Validate) {
    if ($Preferred -and (Test-Path $Preferred)) { return $Preferred }
    foreach ($cmd in @(Get-Command $CommandName -CommandType Application -All -ErrorAction SilentlyContinue)) {
        if ($cmd.Source -and ($cmd.Source -match "\.(exe|cmd|bat)$") -and (Test-Path $cmd.Source) -and (& $Validate $cmd.Source)) {
            return $cmd.Source
        }
    }
    return ""
}

$root = Split-Path $PSScriptRoot -Parent
$installer = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer"
$vs = & "$installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw "Visual Studio with the C++ x64 tools wasn't found." }
$vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"

$src = Join-Path $root "runtime"
$out = Join-Path $src "build"
New-Item -ItemType Directory -Force $out | Out-Null

$cmakeDir = Join-Path $vs "Common7\IDE\CommonExtensions\Microsoft\CMake"
# Skip MSYS/devkitPro cmake on PATH (docs/design.md): native Windows builds advertise Visual Studio + Ninja generators.
$cmake = Find-BuildTool (Join-Path $cmakeDir "CMake\bin\cmake.exe") "cmake" {
    param($p)
    $help = Invoke-ToolProbe $p @("--help")
    ($null -ne $help) -and ($help -match "Visual Studio") -and ($help -match "\bNinja\b")
}
$ninja = Find-BuildTool (Join-Path $cmakeDir "Ninja\ninja.exe") "ninja" {
    param($p)
    $null -ne (Invoke-ToolProbe $p @("--version"))
}
$cfg = if ($Debug) { "Debug" } else { "RelWithDebInfo" }
$sdl = if ($FetchSdl3) { "-DLEONIDA_SDL3_FETCH=ON" } else { "" }
# With VULKAN_SDK set, a configure that still finds no Vulkan headers fails instead of dropping the Vulkan D3D12 device.
$vk = if ($env:VULKAN_SDK) { "-DLEONIDA_REQUIRE_VULKAN=ON" } else { "" }

if ($cmake -and $ninja) {
    $configure = "`"$cmake`" -S `"$src`" -B `"$out`" -G Ninja `"-DCMAKE_MAKE_PROGRAM=$ninja`" -DCMAKE_BUILD_TYPE=$cfg $sdl $vk"
    $build = "`"$cmake`" --build `"$out`" -j $jobs"
    # CALL reparses even quoted arguments and doubles a ^ in a checkout path.
    # A native executable returns directly; a PATH .cmd/.bat wrapper needs its own
    # cmd.exe so it returns here too, without CALL's second expansion.
    if ([System.IO.Path]::GetExtension($cmake) -in @(".cmd", ".bat")) {
        # Escape the outer quotes for the parent batch parser, keeping paths quoted there too.
        $configure = 'cmd.exe /d /s /c ^"' + $configure + '^"'
        $build = 'cmd.exe /d /s /c ^"' + $build + '^"'
    }
    $body = @"
$configure || exit /b 1
$build || exit /b 1
"@
} else {
    # Keep this list in step with runtime\CMakeLists.txt. Optional parts that need a dependency are left out here:
    # the GameInput shim (SDL3), so controllers go through the host's GameInput.dll, if any, as without the shim.
    Write-Warning "CMake and Ninja weren't found (neither VS C++ CMake tools nor native PATH tools): building with cl.exe directly, without the GameInput shim (SDL3). Install CMake + Ninja or VS's CMake tools component to get it."
    $xtaskqueueDir = Join-Path $root "third_party\xtaskqueue"
    $xtaskqueueFiles = "Source\Task\TaskQueue.cpp", "Source\Task\AsyncLib.cpp", "Source\Task\ThreadPool_win32.cpp", "Source\Task\WaitTimer_win32.cpp" | ForEach-Object { "`"$xtaskqueueDir\$_`"" }
    $files = "src\main.cpp", "src\pe_loader.cpp", "shims\registry.cpp", "shims\file_system.cpp", "shims\xmem.cpp", "shims\d3d12_xs.cpp", "shims\d3d12_xs_device.cpp", "shims\d3d12_xs_rootsig.cpp", "shims\xgameruntime.cpp", "shims\xgameruntime_user.cpp", "shims\xgameruntime_gamesave.cpp", "shims\xgameruntime_store.cpp", "shims\xgameruntime_offline.cpp", "shims\winhttp_offline.cpp", "shims\library_loader.cpp", "shims\process.cpp" | ForEach-Object { "`"$src\$_`"" }
    $includes = "/I `"$src\src`" /I `"$src\shims`" /I `"$src\shims\xtaskqueue`" /I `"$xtaskqueueDir\Include`" /I `"$xtaskqueueDir\Source\Task`" /I `"$xtaskqueueDir\Source\Common`""
    $opt = if ($Debug) { "/Od /MDd" } else { "/O2 /MD" }
    $body = @"
cd /d "$out" || exit /b 1
rem BUILD-1457: a game thread with the default stack size gets this program's reserve, and drivers compile on it. The same number as in CMakeLists.txt.
cl /nologo /std:c++20 /EHsc /W4 /permissive- /utf-8 /Zi $opt /MP$jobs /DWIN32_LEAN_AND_MEAN /DNOMINMAX /DUNICODE /D_UNICODE $includes $($files -join ' ') $($xtaskqueueFiles -join ' ') /Fe:gta6rt.exe /link /OPT:NOREF /STACK:8388608 psapi.lib ole32.lib || exit /b 1
cl /nologo /std:c++20 /EHsc /W4 /permissive- /utf-8 /O2 /MT /DWIN32_LEAN_AND_MEAN /DNOMINMAX "$src\src\tls_host.cpp" /LD /Fe:gta6rt_tls_host.dll || exit /b 1
cl /nologo /std:c++20 /EHsc /W4 /permissive- /utf-8 /O2 /DWIN32_LEAN_AND_MEAN /DNOMINMAX /I "$src\shims" "$src\shims\d3d12_xs_rootsig.cpp" "$src\shims\registry.cpp" "$src\shims\d3d12_xs.cpp" "$src\shims\d3d12_xs_device.cpp" "$src\test\root_sig_test.cpp" /Fe:root_sig_test.exe /link d3d12.lib || exit /b 1
cl /nologo /std:c++20 /EHsc /W4 /permissive- /utf-8 /O2 /DWIN32_LEAN_AND_MEAN /DNOMINMAX /I "$src\shims" "$src\shims\d3d12_xs_rootsig.cpp" "$src\shims\registry.cpp" "$src\shims\d3d12_xs.cpp" "$src\shims\d3d12_xs_device.cpp" "$src\test\device_proxy_test.cpp" /Fe:device_proxy_test.exe /link d3d12.lib dxgi.lib || exit /b 1
cl /nologo /std:c++20 /EHsc /W4 /permissive- /utf-8 /O2 /DWIN32_LEAN_AND_MEAN /DNOMINMAX /I "$src\src" "$src\test\descriptor_table_test.cpp" /Fe:descriptor_table_test.exe /link /OPT:NOREF || exit /b 1
cl /nologo /std:c++20 /EHsc /W4 /permissive- /utf-8 /O2 /DWIN32_LEAN_AND_MEAN /DNOMINMAX /I "$src\src" "$src\test\command_list_test.cpp" /Fe:command_list_test.exe /link /OPT:NOREF || exit /b 1
rem BUILD-1457: drivers compile on the checker's threads, and 1 MB of stack is too little for AMD's. The same number as in CMakeLists.txt.
if defined VULKAN_SDK cl /nologo /std:c++20 /EHsc /O2 /DWIN32_LEAN_AND_MEAN /DNOMINMAX /I "%VULKAN_SDK%\Include" "$src\test\vk_pipeline_check.cpp" /Fe:vk_pipeline_check.exe /link /STACK:8388608 || exit /b 1
if defined VULKAN_SDK cl /nologo /std:c++20 /EHsc /O2 /DWIN32_LEAN_AND_MEAN /DNOMINMAX /I "%VULKAN_SDK%\Include" /I "$src\src" "$src\test\vk_texture_test.cpp" /Fe:vk_texture_test.exe || exit /b 1
if defined VULKAN_SDK cl /nologo /std:c++20 /EHsc /O2 /DWIN32_LEAN_AND_MEAN /DNOMINMAX /I "%VULKAN_SDK%\Include" /I "$src\src" "$src\test\vk_sampler_test.cpp" /Fe:vk_sampler_test.exe || exit /b 1
if defined VULKAN_SDK cl /nologo /std:c++20 /EHsc /O2 /DWIN32_LEAN_AND_MEAN /DNOMINMAX /I "%VULKAN_SDK%\Include" /I "$src\src" "$src\test\gfx10_descriptors_test.cpp" /Fe:gfx10_descriptors_test.exe || exit /b 1
if defined VULKAN_SDK cl /nologo /std:c++20 /EHsc /O2 /DWIN32_LEAN_AND_MEAN /DNOMINMAX /I "%VULKAN_SDK%\Include" /I "$src\src" "$src\test\vk_bindless_test.cpp" /Fe:vk_bindless_test.exe || exit /b 1
if defined VULKAN_SDK cl /nologo /std:c++20 /EHsc /O2 /DWIN32_LEAN_AND_MEAN /DNOMINMAX /I "%VULKAN_SDK%\Include" /I "$src\src" "$src\test\vk_descriptor_bench.cpp" /Fe:vk_descriptor_bench.exe || exit /b 1
rem The GPU test harnesses check.py and run_gpu_tests.py require and the fuzzers run (tests\test_build_runtime_fallback.py).
if defined VULKAN_SDK cl /nologo /std:c++20 /EHsc /O2 /DWIN32_LEAN_AND_MEAN /DNOMINMAX /I "%VULKAN_SDK%\Include" /I "$src\src" "$src\test\vk_harness.cpp" /Fe:vk_harness.exe || exit /b 1
if defined VULKAN_SDK cl /nologo /std:c++20 /EHsc /O2 /DWIN32_LEAN_AND_MEAN /DNOMINMAX /I "%VULKAN_SDK%\Include" /I "$src\src" "$src\test\vk_compute_test.cpp" /Fe:vk_compute_test.exe || exit /b 1
if defined VULKAN_SDK cl /nologo /std:c++20 /EHsc /O2 /DWIN32_LEAN_AND_MEAN /DNOMINMAX /I "%VULKAN_SDK%\Include" /I "$src\src" "$src\test\vk_store_test.cpp" /Fe:vk_store_test.exe || exit /b 1
if defined VULKAN_SDK cl /nologo /std:c++20 /EHsc /O2 /DWIN32_LEAN_AND_MEAN /DNOMINMAX /I "%VULKAN_SDK%\Include" /I "$src\src" "$src\test\vk_image_fuzz.cpp" /Fe:vk_image_fuzz.exe || exit /b 1
"@
}

# A .cmd file avoids cmd.exe's quote stripping on paths with spaces.
# Switch to UTF-8 (CP 65001) before invoking CMake/Ninja/cl.exe so non-English MSVC /showIncludes prefixes
# (e.g. Spanish "Nota: inclusión del archivo:") match Ninja's msvc_deps_prefix instead of being mojibaked in OEM CP.
$bat = Join-Path $out "_build.cmd"
# Keep the paths encoded for CP 65001 and start with an ASCII command that cmd.exe can read before switching.
# Here-strings follow the script's checkout line endings; cmd.exe needs CRLF in the generated UTF-8 batch.
$body = $body -replace "`r?`n", "`r`n"
$batchText = "@echo off`r`nchcp 65001 >nul`r`nset `"PATH=$installer;%PATH%`"`r`ncall `"$vcvars`" >nul || exit /b 1`r`n$body`r`n"
[System.IO.File]::WriteAllText($bat, $batchText, [System.Text.UTF8Encoding]::new($false))

# -NoNewWindow attaches cmd/cmake to the parent console, so "Configuring done" / ninja status paint
# over make_game.py's LEONIDA board. Hidden + file redirects keep the console clean; we then emit the
# logs on our stdout so the board's pipe capture still sees them.
# /s with a second pair of quotes: cmd.exe strips only the outer pair, so a path holding & or ^ stays quoted.
$stdoutLog = Join-Path $out "_build.stdout.log"
$stderrLog = Join-Path $out "_build.stderr.log"
foreach ($log in @($stdoutLog, $stderrLog)) {
    if (Test-Path -LiteralPath $log) { Remove-Item -LiteralPath $log -Force }
}
$p = Start-Process -FilePath cmd.exe -ArgumentList "/s", "/c", "`"`"$bat`"`"" -PassThru `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog
$null = $p.Handle  # keeps the exit code readable after WaitForExit
try { $p.PriorityClass = "BelowNormal" } catch {}

function Emit-NewLogLines {
    param([ref]$Offset, [string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return }
    $fs = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read,
        [System.IO.FileShare]::ReadWrite)
    try {
        if ($fs.Length -lt $Offset.Value) { $Offset.Value = 0 }
        [void]$fs.Seek($Offset.Value, [System.IO.SeekOrigin]::Begin)
        $sr = New-Object System.IO.StreamReader($fs)
        while (($line = $sr.ReadLine()) -ne $null) {
            Write-Output $line
        }
        # Keep a partial trailing line for the next poll (ReadLine consumed only complete lines).
        $Offset.Value = $fs.Position
    } finally {
        $fs.Dispose()
    }
}

$outOff = 0
$errOff = 0
while (-not $p.HasExited) {
    Emit-NewLogLines -Offset ([ref]$outOff) -Path $stdoutLog
    Emit-NewLogLines -Offset ([ref]$errOff) -Path $stderrLog
    Start-Sleep -Milliseconds 150
}
Emit-NewLogLines -Offset ([ref]$outOff) -Path $stdoutLog
Emit-NewLogLines -Offset ([ref]$errOff) -Path $stderrLog
if ($p.ExitCode -ne 0) { throw "build failed ($($p.ExitCode))" }
Write-Output "built $out\gta6rt.exe"
