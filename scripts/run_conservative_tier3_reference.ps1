param(
    [string]$Device = ""
)
$ErrorActionPreference = "Stop"

$root = Split-Path $PSScriptRoot -Parent
$runtime = Join-Path $root "runtime"
$build = Join-Path $runtime "build\tier3-reference"

& cmake -S $runtime -B $build -G "Visual Studio 17 2022" -A x64 -DLEONIDA_VULKAN_HEADERS_FETCH=ON
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed ($LASTEXITCODE)." }

& cmake --build $build --config RelWithDebInfo --target d3d12_conservative_tier3_reference_test
if ($LASTEXITCODE -ne 0) { throw "Building the Tier 3 reference test failed ($LASTEXITCODE)." }

$previousFilter = $env:LEONIDA_VK_DEVICE
try {
    if ([string]::IsNullOrWhiteSpace($Device)) {
        if (Test-Path Env:LEONIDA_VK_DEVICE) { Remove-Item Env:LEONIDA_VK_DEVICE }
    } else {
        $env:LEONIDA_VK_DEVICE = $Device
    }
    & ctest --test-dir $build -C RelWithDebInfo -R d3d12_conservative_tier3_reference --output-on-failure -V
    if ($LASTEXITCODE -ne 0) { throw "The Tier 3 reference test failed ($LASTEXITCODE)." }
} finally {
    if ($null -eq $previousFilter) {
        if (Test-Path Env:LEONIDA_VK_DEVICE) { Remove-Item Env:LEONIDA_VK_DEVICE }
    } else {
        $env:LEONIDA_VK_DEVICE = $previousFilter
    }
}
