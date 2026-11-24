param(
  [string]$Output = '',
  [string[]]$Extra = @()
)
# Renders every scene into $Output (default: <repo>/tools/conformance/native/out/run1).
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrEmpty($Output)) { $Output = Join-Path $root 'out/run1' }
$exe = Join-Path $root 'build/Release/conform_native.exe'
if (Test-Path $Output) { Remove-Item $Output -Recurse -Force }
& $exe $Output @Extra
Write-Output "EXITCODE $LASTEXITCODE"
