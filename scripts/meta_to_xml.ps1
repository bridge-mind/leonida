# Converts carved PSO ('PSIN') and RBF ('RBF0') metadata blobs to XML with CodeWalker.Core's parsers.
# Run in Windows PowerShell 5.1.
#   .\meta_to_xml.ps1 -Carve <work>\carve -Out <work>\meta [-Limit N]
# Each blob becomes <Out>\<archive>\<offset>.xml. Failures go to <Out>\failures.txt (UTF-8 with BOM) with the error message.
param(
    [string]$Carve = "<work>\carve",
    [string]$Out = "<work>\meta",
    [string]$CodeWalker = "<ref>\codewalker-gen9",
    [int]$Limit = 0
)
$ErrorActionPreference = "Stop"
try { (Get-Process -Id $PID).PriorityClass = "BelowNormal" } catch {}
Add-Type -Path "$CodeWalker\SharpDX.dll"; Add-Type -Path "$CodeWalker\SharpDX.Mathematics.dll"
[void][Reflection.Assembly]::LoadFrom("$CodeWalker\CodeWalker.Core.dll")
New-Item -ItemType Directory -Force $Out | Out-Null
$fail = Join-Path $Out "failures.txt"
"" | Set-Content -LiteralPath $fail -Encoding utf8
$ok = @{ pso = 0; rbf = 0 }; $bad = @{ pso = 0; rbf = 0 }; $n = 0

foreach ($dir in Get-ChildItem -LiteralPath $Carve -Directory) {
    $idx = Join-Path $dir.FullName "index.json"
    if (-not (Test-Path -LiteralPath $idx)) { continue }
    $files = (Get-Content -LiteralPath $idx -Raw | ConvertFrom-Json).files |
        Where-Object { $_.head.StartsWith("5053494e") -or $_.head.StartsWith("52424630") }
    if (-not $files) { continue }
    $dest = Join-Path $Out $dir.Name
    New-Item -ItemType Directory -Force $dest | Out-Null
    foreach ($r in $files) {
        if ($Limit -and $n -ge $Limit) { break }
        $n++
        $name = "{0:x12}" -f [int64]$r.off
        $kind = if ($r.head.StartsWith("5053494e")) { "pso" } else { "rbf" }
        try {
            $bytes = [IO.File]::ReadAllBytes((Join-Path $dir.FullName "$name.bin"))
            if ($kind -eq "pso") {
                $f = New-Object CodeWalker.GameFiles.PsoFile; $f.Load($bytes)
                $xml = [CodeWalker.GameFiles.PsoXml]::GetXml($f)
            } else {
                $f = New-Object CodeWalker.GameFiles.RbfFile; $f.Load($bytes)
                $xml = [CodeWalker.GameFiles.RbfXml]::GetXml($f)
            }
            [IO.File]::WriteAllText((Join-Path $dest "$name.$kind.xml"), $xml)
            $ok[$kind]++
        } catch {
            $bad[$kind]++
            $msg = $_.Exception.InnerException; if (-not $msg) { $msg = $_.Exception }
            "$($dir.Name)\$name $kind : $($msg.Message)" | Add-Content -LiteralPath $fail -Encoding utf8
        }
    }
    if ($Limit -and $n -ge $Limit) { break }
}
"pso ok $($ok.pso) failed $($bad.pso); rbf ok $($ok.rbf) failed $($bad.rbf) -> $Out"
