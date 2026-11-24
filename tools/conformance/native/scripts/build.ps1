# Builds conform_native. Run from anywhere:
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\conformance\native\scripts\build.ps1
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$cmake = (Get-Command cmake -ErrorAction SilentlyContinue)
if ($cmake) { $cmake = $cmake.Path } else { $cmake = 'C:\Program Files\CMake\bin\cmake.exe' }
$source = $root
$build = Join-Path $root 'build'

& $cmake -S $source -B $build -G 'Visual Studio 17 2022' -A x64
if ($LASTEXITCODE -ne 0) { Write-Output "CONFIGURE EXIT $LASTEXITCODE"; exit $LASTEXITCODE }

& $cmake --build $build --config Release --parallel
$buildExit = $LASTEXITCODE
Write-Output "BUILD EXIT $buildExit"
exit $buildExit
