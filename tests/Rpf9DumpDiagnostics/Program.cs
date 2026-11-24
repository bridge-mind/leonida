using System.Buffers.Binary;
using RpfPreview.Core;

Expect([], 4, "unrecognized-header");
Expect([], RageModelDecoder.MaxGraphicsBytes + 1, "graphics-size-limit");

// A synthetically truncated resource prefix containing a system-page pointer.
var prefix = new byte[16];
BinaryPrimitives.WriteUInt64LittleEndian(prefix.AsSpan(8), 0x50000080);
Expect(prefix, 16, "truncated-system");

// Recognizable drawable header with a deliberately malformed page table.
var header = new byte[0x100];
foreach (int at in new[] { 8, 16, 80, 168 })
    BinaryPrimitives.WriteUInt64LittleEndian(header.AsSpan(at), 0x500000c0);
BinaryPrimitives.WriteSingleLittleEndian(header.AsSpan(44), 2);
foreach (int at in new[] { 64, 68, 72 }) BinaryPrimitives.WriteSingleLittleEndian(header.AsSpan(at), 1);
Expect(header, 16, "invalid-or-unsupported-layout");
Console.WriteLine("Model decode diagnostic smoke tests passed.");

// Util.WriteAtomic: outputs appear under their final name only once complete.
string dir = Directory.CreateTempSubdirectory("rpf9dump-test").FullName;
try
{
    string path = Path.Combine(dir, "a.dds");
    Util.WriteAllBytes(path, [1, 2, 3]);
    Check(File.ReadAllBytes(path).SequenceEqual(new byte[] { 1, 2, 3 }), "written file has the data");
    Check(Directory.GetFiles(dir).Length == 1, "no .partial left after a write");

    Util.WriteAllText(path, "new");
    Check(File.ReadAllText(path) == "new", "an existing file is replaced");

    string failed = Path.Combine(dir, "b.dds");
    try { Util.WriteAtomic(failed, t => { File.WriteAllBytes(t, [9]); throw new IOException("disk full"); }); Check(false, "error propagates"); }
    catch (IOException) { }
    Check(!File.Exists(failed) && Directory.GetFiles(dir, "b.dds*partial").Length == 0,
        "a failed write leaves neither file nor a .partial");

    // A stopped run's leftover .partial must not block a writer that opens with FileMode.CreateNew.
    string stale = Path.Combine(dir, "c.xma.wav");
    File.WriteAllBytes(stale + ".partial", [7]);
    Util.WriteAtomic(stale, t => { using var f = new FileStream(t, FileMode.CreateNew); f.WriteByte(5); });
    Check(File.ReadAllBytes(stale).SequenceEqual(new byte[] { 5 })
         && !File.Exists(stale + ".partial")
         && Directory.GetFiles(dir, "c.xma.wav*partial").Length == 0,
         "stale .partial is replaced");
}
finally { Directory.Delete(dir, true); }
Console.WriteLine("Atomic write tests passed.");

// Scan counts round-trip through the index cache's coverage text, including older culture-formatted caches.
Check(ScanCounts.Candidates(ScanCounts.Coverage(183801, 149811)) == 183801, "invariant coverage");
Check(ScanCounts.Candidates("rpf9dump full scan, 183,801 candidates, 149,811 recovered") == 183801, "comma-grouped coverage");
Check(ScanCounts.Candidates("rpf9dump full scan, 183\u00A0801 candidates, 149\u00A0811 recovered") == 183801, "space-grouped coverage");
// Every culture's N0 output must read back (de-CH uses U+2019, ar-* U+066C, nqo U+060C, ff-Adlm U+2E41).
foreach (var culture in System.Globalization.CultureInfo.GetCultures(System.Globalization.CultureTypes.AllCultures))
{
    string old = string.Format(culture, "rpf9dump full scan, {0:N0} candidates, {1:N0} recovered", 183801, 149811);
    Check(ScanCounts.Candidates(old) == 183801, "coverage written under " + culture.Name);
}
Check(ScanCounts.Candidates("App scan of the first 16 MiB") is null, "coverage written by another tool");
Check(ScanCounts.Candidates(null) is null, "no cache");
Console.WriteLine("Scan count tests passed.");

// Util.MeshKey reads the model/geometry pair the decoder puts in a mesh name; the geometry index of a dehydrated
// BLAS leaf counts geometries inside one model, so the pair decides which list a signature is matched against.
Check(Util.MeshKey("model_0_geometry_3") == (0, 3), "mesh key model 0 geometry 3");
Check(Util.MeshKey("model_12_geometry_0") == (12, 0), "mesh key model 12 geometry 0");
Check(Util.MeshKey("model_7_geometry_11") == (7, 11), "mesh key two-digit geometry");
foreach (string other in new[] { "", "model", "model_0_geometry", "model_0_geometry_1_2", "Model_0_geometry_1",
             "model_x_geometry_1", "model_0_geometry_1a", "model__geometry_1", "model_0_Geometry_1", "model_-1_geometry_1" })
    Check(Util.MeshKey(other) == (-1, -1), "not a mesh key: " + other);
Console.WriteLine("Mesh key tests passed.");

// RecoveredResource.Read turns the recovery step's block list back into the resource's bytes: a window slice has no
// archive document, so this is the only reader models and textures have there.
{
    byte[] system = System.Text.Encoding.ASCII.GetBytes("system segment, zlib");
    var gfx = new byte[64];
    for (int i = 0; i < gfx.Length; i++) gfx[i] = (byte)(0x40 + i);
    using var compressed = new MemoryStream();
    using (var deflate = new System.IO.Compression.ZLibStream(compressed, System.IO.Compression.CompressionLevel.SmallestSize, true))
        deflate.Write(system);
    byte[] zblock = compressed.ToArray();

    string segmentDir = Directory.CreateTempSubdirectory("rpf9dump-segments").FullName;
    try
    {
        string path = Path.Combine(segmentDir, "window.bin");
        var file = new byte[32 + zblock.Length + 16 + gfx.Length];
        Array.Fill(file, (byte)0xAA);
        zblock.CopyTo(file, 32);
        gfx.CopyTo(file, 32 + zblock.Length + 16);
        File.WriteAllBytes(path, file);
        ArchiveEntry Entry(params ArchiveSegment[] segments) => new("recovered/models/x_at_0.r9modeldata",
            segments.Sum(s => s.StoredSize), segments.Sum(s => s.Size), 32, false, false, false, "Recovered model", 0) { Segments = segments };
        var entry = Entry(new ArchiveSegment(32, zblock.Length, system.Length),
                          new ArchiveSegment(32 + zblock.Length + 16, gfx.Length, gfx.Length) { IsRaw = true });
        byte[] read = RecoveredResource.Read(path, entry);
        Check(read.SequenceEqual(system.Concat(gfx)), "one compressed and one raw block, in stored order");

        // A block that reaches past the file is an error, not a short read.
        var pastEnd = Entry(new ArchiveSegment(32, file.Length, system.Length));
        try { RecoveredResource.Read(path, pastEnd); Check(false, "block past the end is rejected"); }
        catch (InvalidDataException) { }
        // A raw block reads its own bytes; the 6-byte minimum is the smallest block the recovery step records.
        var shortRaw = Entry(new ArchiveSegment(32 + zblock.Length + 16, 8, 8) { IsRaw = true });
        Check(RecoveredResource.Read(path, shortRaw).SequenceEqual(gfx[..8]), "a raw block reads its own bytes");
        var tooSmall = Entry(new ArchiveSegment(32 + zblock.Length + 16, 4, 4) { IsRaw = true });
        try { RecoveredResource.Read(path, tooSmall); Check(false, "a block under six bytes is rejected"); }
        catch (InvalidDataException) { }
        // An entry that records no blocks is rejected rather than returning an empty resource.
        var oversize = new ArchiveEntry("recovered/models/x.r9modeldata", 1, 1, 32, false, false, false, "Recovered model", 0);
        try { RecoveredResource.Read(path, oversize); Check(false, "an entry without blocks is rejected"); }
        catch (InvalidDataException) { }
    }
    finally { Directory.Delete(segmentDir, true); }
}
Console.WriteLine("Recovered segment tests passed.");

// Only a full scan replaces rpf9dump's own: rpf9-explorer saves partial model indexes while indexing and on cancel.
Check(!ScanCounts.IsFullScan(Index("Dictionary-aware model scan in progress \u00B7 1,001/183,801 containers validated.", false)), "GUI index saved while indexing");
Check(!ScanCounts.IsFullScan(Index("Dictionary-aware model scan canceled. Validated models are retained; discovery will resume on reopen.", false)), "GUI index saved on cancel");
Check(ScanCounts.IsFullScan(Index("Full 1,234,567-byte model-container scan \u00B7 149,811 supported models including drawable dictionary children.", true)), "GUI index marked complete");
Check(ScanCounts.IsFullScan(Index(ScanCounts.Coverage(183801, 149811), true)), "rpf9dump index");
Check(ScanCounts.IsFullScan(Index(ScanCounts.Coverage(183801, 149811), false)), "rpf9dump index saved without the complete mark");
Check(ScanCounts.IsFullScan(Index(string.Format(System.Globalization.CultureInfo.GetCultureInfo("de-CH"), "rpf9dump full scan, {0:N0} candidates, {1:N0} recovered", 183801, 149811), false)), "culture-formatted rpf9dump index");
Check(!ScanCounts.IsFullScan(null), "no cached index");
Console.WriteLine("Cached index tests passed.");

static void Check(bool ok, string what)
{
    if (!ok) throw new Exception($"Failed: {what}");
}

static RecoveryIndexCache.CachedIndex Index(string coverage, bool complete) =>
    new("rpfpreview-payload-index/models/4", "models", "levels.rpf", 0, 0, "", coverage, []) { IsComplete = complete };

static void Expect(ReadOnlySpan<byte> system, int graphicsBytes, string expected)
{
    string actual = ModelDecodeFailure.Explain(system, graphicsBytes).Code;
    if (actual != expected) throw new Exception($"Expected {expected}, got {actual}");
}
