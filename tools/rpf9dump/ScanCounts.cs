using System.Text.RegularExpressions;
using RpfPreview.Core;

// Scan-level counts for the manifest. Candidates that RecoverAtOffsets rejects never reach Program.cs, so
// Candidates - Recovered is the only place those rejects show up.
internal static class ScanCounts
{
    // Written into the index cache's coverage text on a full scan.
    public static string Coverage(int candidates, int recovered) =>
        FormattableString.Invariant($"rpf9dump full scan, {candidates} candidates, {recovered} recovered");

    // Reads the candidate count back from a cached index. Older caches wrote culture-formatted numbers
    // ("183,801", "183'801", "183\u2019801", ...), so any punctuation or space between the digits is dropped. Null when the text is not ours or not readable.
    public static int? Candidates(string? coverage)
    {
        if (coverage is null) return null;
        var m = Regex.Match(coverage, @"^rpf9dump full scan, ([0-9][0-9\p{P}\p{Zs}]*) candidates,");
        if (!m.Success) return null;
        string digits = new(m.Groups[1].Value.Where(char.IsAsciiDigit).ToArray());
        return int.TryParse(digits, out int n) ? n : null;
    }

    // Whether a cached index can stand in for a full scan. rpf9-explorer saves its model index while it is still
    // indexing and when the scan is cancelled, and marks it complete only at the end. rpf9dump before this check saved
    // full scans without the mark, so its own coverage text counts too.
    public static bool IsFullScan(RecoveryIndexCache.CachedIndex? cached) =>
        cached is not null && (cached.IsComplete || Candidates(cached.Coverage) is not null);
}
