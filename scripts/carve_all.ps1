# Carves every .rpf in the game folder into <Out>\<archive path with _>, smallest first (research tool; the
# build itself doesn't need it). Skips archives that already have an index.json. Summary: <Out>\coverage.json
param([Parameter(Mandatory)][string]$Game, [Parameter(Mandatory)][string]$Out)
$root = Split-Path $PSScriptRoot -Parent
$tool = Join-Path $root "tools\rpf9_carve.py"
# -Game may be relative or end in a separator; archive names are cut from a separator-ended prefix.
$Game = (Resolve-Path -LiteralPath $Game).ProviderPath
# Resolve-Path preserves 8.3 short components (e.g. RUNNER~1), while Get-ChildItem emits canonical
# FullName paths. Deriving the prefix from Resolve-Path then cuts the wrong number of characters and
# re-carves already-caved archives; use the canonical form for both sides of the Substring.
$prefix = (Get-Item -LiteralPath $Game).FullName.TrimEnd('\') + '\'
$destinations = [System.Collections.Generic.Dictionary[string, string]]::new([StringComparer]::OrdinalIgnoreCase)
# Validate every destination before accepting any cached index or starting a carve.
$archives = foreach ($f in Get-ChildItem -LiteralPath $Game -Recurse -Filter *.rpf | Sort-Object Length) {
    $rel = $f.FullName.Substring($prefix.Length)
    $name = [IO.Path]::GetFileNameWithoutExtension(($rel -replace "[\\/]", "_"))
    if ($destinations.ContainsKey($name)) {
        throw ("Archives '{0}' and '{1}' both map to '{2}'." -f $destinations[$name], $rel, $name)
    }
    $destinations.Add($name, $rel)
    [pscustomobject]@{ file = $f; relative = $rel; destination = Join-Path $Out $name }
}

$rows = @()
foreach ($archive in $archives) {
    $f = $archive.file
    $rel = $archive.relative
    $dest = $archive.destination
    $index = Join-Path $dest "index.json"
    if (-not (Test-Path -LiteralPath $index)) {
        Write-Host ("{0:HH:mm:ss} carving {1} ({2:N1} GiB)" -f (Get-Date), $rel, ($f.Length / 1GB))
        python -I $tool $f.FullName $dest | Select-Object -Last 1
        if ($LASTEXITCODE -ne 0) { throw "Carving $rel failed (exit $LASTEXITCODE)" }
    }
    try {
        $s = (Get-Content -LiteralPath $index -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop).summary
        foreach ($field in 'files', 'bink2_files', 'span', 'compressed_bytes', 'unread_bytes', 'inflated_bytes') {
            $value = $s.$field
            if ((($value -isnot [int]) -and ($value -isnot [long])) -or $value -lt 0) {
                throw "summary.$field must be a non-negative integer."
            }
        }
        if ($s.span -eq 0 -and ($s.compressed_bytes -ne 0 -or $s.unread_bytes -ne 0)) {
            throw 'A zero summary.span requires zero compressed and unread bytes.'
        }
        # Stored streams and unread gaps occupy disjoint parts of the span. Subtract instead of
        # adding the counters so a total just above Int64.MaxValue cannot lose precision.
        if ($s.compressed_bytes -gt $s.span -or $s.unread_bytes -gt ($s.span - $s.compressed_bytes)) {
            throw 'summary.compressed_bytes and summary.unread_bytes must fit within summary.span.'
        }
        if ($s.bink2_files -gt $s.files) {
            throw 'summary.bink2_files must not exceed summary.files.'
        }
    } catch {
        throw "Invalid carve index '$index': $($_.Exception.Message)"
    }
    $denominator = [math]::Max(1L, $s.span)
    $rows += [pscustomobject]@{ archive = $rel; files = $s.files; bink2 = $s.bink2_files; span = $s.span
                               readable = [math]::Round($s.compressed_bytes / $denominator, 4)
                               unread = [math]::Round($s.unread_bytes / $denominator, 4); inflated = $s.inflated_bytes }
}
$rows | ConvertTo-Json | Set-Content -Encoding utf8 -LiteralPath (Join-Path $Out "coverage.json")
$rows | Format-Table archive, files, bink2, @{n = "GiB"; e = { [math]::Round($_.span / 1GB, 2) } }, readable, unread -AutoSize | Out-String -Width 200
