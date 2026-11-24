$ErrorActionPreference = 'Stop'
$source = Split-Path -Parent $PSScriptRoot
$scratch = Join-Path $source 'negative-control'
$cmake = (Get-Command cmake -ErrorAction SilentlyContinue)
if ($cmake) { $cmake = $cmake.Path } else { $cmake = 'C:\Program Files\CMake\bin\cmake.exe' }

# This script exists to prove the scenes are not vacuous: it disables three pieces of
# pipeline state and requires the suite to notice exactly those three scenes. It asserts
# its own result, so a silent no-op edit cannot be mistaken for evidence.

function Replace-ExactlyOnce {
    param([string] $Text, [string] $Search, [string] $Replacement, [string] $What)
    $count = ([regex]::Matches($Text, [regex]::Escape($Search))).Count
    if ($count -ne 1) {
        throw "expected exactly one occurrence of '$Search' while disabling $What, found $count. The source moved; update negative-control.ps1 before trusting this test."
    }
    return $Text.Replace($Search, $Replacement)
}

Write-Output '=== building a throwaway copy with depth, blend and scissor state disabled ==='
if (Test-Path $scratch) { Remove-Item $scratch -Recurse -Force }
New-Item -ItemType Directory "$scratch\src" | Out-Null
Copy-Item "$source\CMakeLists.txt" $scratch
Copy-Item "$source\src\*" "$scratch\src"

# 1. depth test off in the vertex-colour pipeline
$renderer = Get-Content "$scratch\src\renderer.cpp" -Raw
$renderer = Replace-ExactlyOnce $renderer 'description.DepthStencilState.DepthEnable = TRUE;' 'description.DepthStencilState.DepthEnable = FALSE;' 'the depth test'
# 2. blending off
$renderer = Replace-ExactlyOnce $renderer 'description.BlendState.RenderTarget[0].BlendEnable = TRUE;' 'description.BlendState.RenderTarget[0].BlendEnable = FALSE;' 'blending'
Set-Content -Path "$scratch\src\renderer.cpp" -Value $renderer -NoNewline

# 3. scissor rect widened to the whole target
$plan = Get-Content "$scratch\src\scene_plan.cpp" -Raw
$plan = Replace-ExactlyOnce $plan 'plan.scissor = {conform::kScissorLeft, conform::kScissorTop, conform::kScissorRight,' 'plan.scissor = {0, 0, conform::kSceneWidth,' 'the scissor rectangle'
$plan = Replace-ExactlyOnce $plan 'conform::kScissorBottom};' 'conform::kSceneHeight};' 'the scissor rectangle bottom edge'
Set-Content -Path "$scratch\src\scene_plan.cpp" -Value $plan -NoNewline

& $cmake -S $scratch -B "$scratch\build" -G 'Visual Studio 17 2022' -A x64 | Out-Null
& $cmake --build "$scratch\build" --config Release --parallel | Select-Object -Last 2
if ($LASTEXITCODE -ne 0) { Write-Output "BUILD FAILED $LASTEXITCODE"; exit 1 }

Write-Output '=== run: which scenes notice? ==='
$output = & "$scratch\build\Release\conform_native.exe" (Join-Path $source 'out/negative-control') 2>&1
$exit = $LASTEXITCODE
$output | Select-String -Pattern '^FAIL|^readback checks|^RESULT'
Write-Output "EXITCODE $exit"

# Assert, rather than print and hope.
$failedScenes = $output |
    Select-String -Pattern '^FAIL (?:readback|png) scene=(\w+)' |
    ForEach-Object { $_.Matches[0].Groups[1].Value } |
    Sort-Object -Unique
$expected = @('blend', 'depth_order', 'scissor')
if (Compare-Object $failedScenes $expected) {
    Write-Output "NEGATIVE CONTROL FAILED: scenes that noticed = $($failedScenes -join ', '); expected $($expected -join ', ')"
    exit 1
}
if ($exit -eq 0) {
    Write-Output 'NEGATIVE CONTROL FAILED: the corrupted build still exited 0'
    exit 1
}
Write-Output "NEGATIVE CONTROL PASSED: exactly $($expected -join ', ') failed"
Remove-Item $scratch -Recurse -Force
exit 0
