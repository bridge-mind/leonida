// Bulk exporter around rpf9-explorer's Core (github.com/breadbbbb/rpf9-explorer).
// Local use only: writes OBJ/DDS/JSON from the user's own package. No keys, no TOC decryption.
using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using RpfPreview.Core;

// Optional --jobs N anywhere among the args (default: every core). Stripped before mode/src/outdir.
var argv = args.ToList();
int jobs = Environment.ProcessorCount;
for (int i = 0; i < argv.Count; )
{
    if (argv[i] == "--jobs")
    {
        if (i + 1 >= argv.Count || !int.TryParse(argv[i + 1], out jobs) || jobs < 1)
        {
            Console.Error.WriteLine("--jobs must be at least 1");
            return 2;
        }
        argv.RemoveAt(i);
        argv.RemoveAt(i);
        continue;
    }
    i++;
}
// The mode is checked before the source or the output folder is touched. A typo used to fall through every branch:
// it created the output folder, printed ok=0 fail=0 skip=0 and exited 0, which make_game reads as a finished stage.
string[] modes = ["ttpf", "audio", "xma", "models", "textures"];
if (argv.Count < 3) { Usage(); return 2; }
if (!modes.Contains(argv[0])) { Console.Error.WriteLine($"unknown mode '{argv[0]}'"); Usage(); return 2; }
string mode = argv[0], src = Path.GetFullPath(argv[1]), outRoot = Path.GetFullPath(argv[2]);
// --start/--length scan one window of the file instead of the whole archive. A window is a raw slice of an
// archive (`levels.rpf` cut at an offset), so it need not carry the file's header and it is not an
// ArchiveDocument: models and textures then read every resource from its own compression blocks
// (tools/rpf9dump/Segments.cs). A window is how a multi-GiB archive is exported a piece at a time.
long scanStart = 0; long? scanLength = null;
for (int i = 3; i < argv.Count; i += 2)
{
    if (i + 1 >= argv.Count || !long.TryParse(argv[i + 1], NumberStyles.None, CultureInfo.InvariantCulture, out long bytes) || bytes < 0)
    {
        Console.WriteLine($"{argv[i]} needs a non-negative byte count.");
        return 2;
    }
    if (argv[i] == "--start") scanStart = bytes;
    else if (argv[i] == "--length") scanLength = bytes;
    else { Console.WriteLine($"Unknown option {argv[i]}."); return 2; }
}
bool window = scanStart != 0 || scanLength is not null;
if (window && mode is not ("models" or "textures"))
{
    Console.WriteLine("--start/--length apply to models and textures; other modes need a recognized archive.");
    return 2;
}
// A missing source, or a directory passed as one, is a usage error. Check it before the output folder is
// created: FileInfo.Length would otherwise throw after <out>/<name> already exists.
if (!File.Exists(src)) { Console.Error.WriteLine($"source is not a file: {src}"); return 2; }
long length = new FileInfo(src).Length;
if (window && scanStart > length)
{
    Console.Error.WriteLine($"--start ({scanStart}) exceeds source length ({length}).");
    return 2;
}
long? effectiveScanLength = scanLength is long reqSpan ? Math.Min(reqSpan, length - scanStart) : null;
var json = new JsonSerializerOptions { WriteIndented = true };
var sw = Stopwatch.StartNew();
string baseName = Path.GetFileNameWithoutExtension(src);
string outDir = Path.Combine(outRoot, baseName); Directory.CreateDirectory(outDir);
int ok = 0, fail = 0, skip = 0, writeFail = 0;
// Every mode below works on all cores by default. Core's readers open their own read handle per call, entry and
// file names come from offsets and indexes, and results are put back in source order, so outputs match a
// one-thread run. --jobs caps that when make_game.py overlaps this with another CPU-heavy stage.
var par = new ParallelOptions { MaxDegreeOfParallelism = jobs };

if (mode == "ttpf")
{
    var info = TtpfReader.Inspect(src);
    var rows = new object[info.Textures.Count];
    int done = 0;
    Parallel.For(0, info.Textures.Count, par, i =>
    {
        var t = info.Textures[i];
        string name = $"{t.Index:D6}_{Util.Safe(Path.GetFileNameWithoutExtension(t.Name))}.dds";
        string path = Path.Combine(outDir, name);
        string? file = t.CanExtract ? name : null, limitation = t.Limitation;
        if (!t.CanExtract) Interlocked.Increment(ref skip);
        else if (File.Exists(path)) Interlocked.Increment(ref ok);
        else
        {
            try
            {
                // Content the decoder cannot handle is a skip like !CanExtract. Pool read errors, DDS write errors,
                // cancellation and anything unexpected stay failures.
                byte[]? dds = null;
                try { dds = TtpfReader.CreateDds(src, t.Index); }
                catch (Exception e) when (e is NotSupportedException or InvalidDataException)
                { Interlocked.Increment(ref skip); file = null; limitation = e.Message; }
                if (dds != null) { Util.WriteAllBytes(path, dds); Interlocked.Increment(ref ok); }
            }
            catch (Exception e) { if (Interlocked.Increment(ref fail) <= 5) Console.WriteLine($"  fail {t.Index}: {e.GetType().Name} {e.Message}"); }
            int n = Interlocked.Increment(ref done);
            if (n % 5000 == 0) Console.WriteLine($"  {n}/{info.Textures.Count}");
        }
        rows[i] = new { t.Index, t.Name, t.Id, AssetKey = t.AssetKey.ToString("X16"), t.Width, t.Height,
            t.Depth, t.Dimension, t.DxgiFormat, t.Mips, t.SourceMips, t.BaseMipLevel, Limitation = limitation, File = file };
    });
    // The manifest is the build's completion marker. Keep successful DDS for retry, but withdraw a stale marker.
    string manifest = Path.Combine(outDir, "manifest.json");
    if (fail == 0)
        Util.WriteAllText(manifest, JsonSerializer.Serialize(new { src, info.Version, Count = info.Textures.Count, info.Metadata, Textures = rows }, json));
    else
    {
        try { WithdrawMarker(manifest); }
        catch (Exception ex) { Console.WriteLine($"  cannot withdraw the stale manifest: {ex.GetType().Name} {ex.Message}"); return 1; }
    }
}
else if (mode == "audio")
{
    if (!AwcRecovery.HasCompleteCache(src))
    {
        long last = -1;
        var prog = new Progress<long>(p => { long s = p / (4L << 30); if (s != last) { last = s; Console.WriteLine($"  awc scan {100.0 * p / length:0.0}%"); } });
        var found = AwcRecovery.RecoverFull(src, prog);
        AwcRecovery.WriteCache(src, found);
        Console.WriteLine($"  {found.Count:N0} AWC banks ({sw.Elapsed.TotalSeconds:0}s)");
    }
    var doc = ArchiveReader.Open(src);
    var banks = doc.Entries.Where(e => e.Extension == ".awc" && e.CanExtract).ToList();
    string sub = Path.Combine(outDir, "awc");
    var todo = banks.Where(e => !File.Exists(ArchiveReader.SafeDestination(sub, e.Path))).ToList();
    Console.WriteLine($"  {banks.Count:N0} banks, {todo.Count:N0} to write");
    for (int i = 0; i < todo.Count; i += 500)
        doc.ExtractAsync(todo.Skip(i).Take(500), sub).GetAwaiter().GetResult();
    ok = banks.Count;
    var rows = banks.AsParallel().AsOrdered().WithDegreeOfParallelism(par.MaxDegreeOfParallelism).Select(e => { var info = AwcInfo.Inspect(src, e.Offset, e.Size);
        return new { e.Path, e.Offset, e.Size, Streams = info?.Streams.Select(s => new { s.Index, Id = s.Id.ToString("X8"), s.Codec, s.Channels, s.SampleRate, s.DurationSeconds, s.IsStreamed }) }; }).ToList();
    Util.WriteAllText(Path.Combine(outDir, "audio_manifest.json"), JsonSerializer.Serialize(new { src, Count = banks.Count, Banks = rows }, json));
}
else if (mode == "xma")
{
    // XMA2 streams (AWC codec 5) as RIFF XMA2 .wav files that ffmpeg decodes (AwcRecovery.WriteXmaWave).
    var doc = ArchiveReader.Open(src);
    var banks = doc.Entries.Where(e => e.Extension == ".awc" && e.CanExtract).ToList();
    string sub = Path.Combine(outDir, "xma"); Directory.CreateDirectory(sub);
    Parallel.ForEach(banks, par, e =>
    {
        var info = AwcInfo.Inspect(src, e.Offset, e.Size);
        if (info == null) return;
        foreach (var st in info.Streams.Where(x => x.Codec == 5))
        {
            string dir = Path.Combine(sub, Util.Safe(Path.GetFileNameWithoutExtension(e.Name)));
            Directory.CreateDirectory(dir);
            string path = Path.Combine(dir, $"{st.Id:X8}.xma.wav");
            if (File.Exists(path)) { Interlocked.Increment(ref ok); continue; }
            try { Util.WriteAtomic(path, tmp => AwcRecovery.WriteXmaWave(src, e.Offset, e.Size, st.Index, tmp)); Interlocked.Increment(ref ok); }
            catch (Exception ex) { if (Interlocked.Increment(ref fail) <= 5) Console.WriteLine($"  fail {e.Name}#{st.Index}: {ex.GetType().Name} {ex.Message}"); }
        }
    });
    // make_game marks audio/xma done on exit 0, so a failed stream must fail the run to be retried.
    Console.WriteLine($"{mode} {baseName}: ok={ok} fail={fail} skip={skip} in {sw.Elapsed.TotalSeconds:0}s");
    return fail != 0 ? 1 : 0;
}
else if (mode is "models" or "textures")
{
    // Neither mode needs the archive document any more: the entries come from the index cache or from this run's
    // scan, and every resource is read from its own compression blocks (tools/rpf9dump/Segments.cs). That is what
    // lets a raw slice of an archive - a --start/--length window, or a file cut out of levels.rpf - be exported at
    // all: a slice has no 9FPR header, so there is no document to open.
    List<ArchiveEntry> list;
    int? candidates;
    if (window)
    {
        // A sub-range is never cached: the index cache is keyed by the file, not by the window inside it.
        var cands = Scan(src, scanStart, effectiveScanLength);
        candidates = cands.Count;
        list = Recover(src, length, cands);
        Console.WriteLine("  a window is scanned again on every run: no index cache is written");
    }
    else
    {
        byte[] header; using (var f = File.OpenRead(src)) { header = new byte[(int)Math.Min(64, length)]; f.ReadExactly(header); }
        var cached = RecoveryIndexCache.Load(mode, src, length, header);
        // A partial index (rpf9-explorer stopped or still indexing) would export only part of the archive: scan again.
        if (cached != null && !ScanCounts.IsFullScan(cached)) { Console.WriteLine($"  cached {mode} index is not marked as a full scan ({cached.Entries.Count:N0} entries), rescanning"); cached = null; }
        candidates = ScanCounts.Candidates(cached?.Coverage);
        if (cached == null)
        {
            var cands = Scan(src, 0, null);
            list = Recover(src, length, cands);
            RecoveryIndexCache.Save(mode, src, list, ScanCounts.Coverage(cands.Count, list.Select(e => e.Offset).Distinct().Count()), complete: true);
            candidates = cands.Count;
        }
        else
        {
            string want = mode == "models" ? "Recovered model" : "Recovered texture dictionary";
            list = cached.Entries.Where(e => e.Compression == want).ToList();
        }
    }
    Console.WriteLine($"  {list.Count:N0} {mode} entries ({sw.Elapsed.TotalSeconds:0}s)");
    // A drawable dictionary yields one entry per child, all with the candidate's offset: count resources by offset.
    int recovered = list.Select(e => e.Offset).Distinct().Count();
    if (candidates is int scannedCount && scannedCount > recovered)
        Console.WriteLine($"  {scannedCount - recovered:N0} of {scannedCount:N0} scan candidates were not recovered (rejected before decoding)");

    // Scan a range and recover every resource it holds. Both run on all cores; the result is put back in offset
    // order, so it does not depend on which chunk finished first.
    IReadOnlyList<long> Scan(string path, long start, long? span)
    {
        long total = Math.Max(1, span ?? (length - start)), last = -1;
        var progress = new Progress<long>(p =>
        {
            long percent = 100 * (p - start) / total;
            if (percent != last) { last = percent; Console.WriteLine($"  scan {percent}%"); }
        });
        var found = mode == "models"
            ? Rpf9ModelRecovery.FindCandidateOffsets(path, start, span, progress)
            : Rpf9TextureRecovery.FindCandidateOffsets(path, start, span, progress);
        Console.WriteLine($"  candidates {found.Count:N0} ({sw.Elapsed.TotalSeconds:0}s)");
        return found;
    }

    List<ArchiveEntry> Recover(string path, long fileLength, IReadOnlyList<long> offsets)
    {
        // Indexes are reassigned in offset order below, so the chunks' scheduling does not show in the result.
        var chunks = offsets.Chunk(32).ToArray();
        var got = new IReadOnlyList<ArchiveEntry>[chunks.Length];
        int validated = 0, found = 0;
        Parallel.For(0, chunks.Length, par, i =>
        {
            got[i] = mode == "models" ? Rpf9ModelRecovery.RecoverAtOffsets(path, fileLength, chunks[i]) : Rpf9TextureRecovery.RecoverAtOffsets(path, fileLength, chunks[i]);
            Interlocked.Add(ref found, got[i].Count);
            int n = Interlocked.Add(ref validated, chunks[i].Length);
            if (n / 6400 != (n - chunks[i].Length) / 6400) Console.WriteLine($"  validated {n:N0}/{offsets.Count:N0} -> {Volatile.Read(ref found):N0} ({sw.Elapsed.TotalSeconds:0}s)");
        });
        return got.SelectMany(g => g).OrderBy(e => e.Offset).Select((e, k) => e with { Index = k }).ToList();
    }
    string sub = Path.Combine(outDir, mode); Directory.CreateDirectory(sub);
    // The manifest is streamed row by row: holding 150k rows (each capturing its decoded model through lazy
    // projections) ran out of memory on levels.rpf. Written to a .partial file and renamed when complete.
    string manifestPath = Path.Combine(sub, "manifest.json"), partial = manifestPath + ".partial";
    using var mstream = new FileStream(partial, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20);
    using var writer = new Utf8JsonWriter(mstream, new JsonWriterOptions { Indented = true });
    writer.WriteStartObject();
    writer.WriteString("src", src);
    writer.WriteNumber("Count", list.Count);
    // Scan-level counts. Candidates the recovery step rejects never reach the per-item loop or
    // ModelDecodeFailures, so NotRecovered is the only place they show up. Recovered counts resources (candidate
    // offsets); Entries counts manifest items (one per dictionary child). Null when a cached index does not record
    // the candidate count.
    writer.WriteStartObject("Scan");
    if (candidates is int scanned) writer.WriteNumber("Candidates", scanned); else writer.WriteNull("Candidates");
    writer.WriteNumber("Recovered", recovered);
    if (candidates is int total) writer.WriteNumber("NotRecovered", total - recovered); else writer.WriteNull("NotRecovered");
    writer.WriteNumber("Entries", list.Count);
    if (window)
    {
        // Which slice of the file this manifest covers, so a later run can name the same window.
        writer.WriteNumber("Start", scanStart);
        writer.WriteNumber("Length", effectiveScanLength ?? (length - scanStart));
    }
    writer.WriteEndObject();
    writer.WriteStartArray("Items");
    void Row(object row) => JsonSerializer.Serialize(writer, row, json);
    int embedded = 0;
    var modelDecodeFailures = new SortedDictionary<string, int>(StringComparer.Ordinal);
    // Entries are decoded and written in parallel a batch at a time; rows, counters and failure messages are then
    // taken in list order on this thread, so the manifest and the log read as a one-thread run. The batch bounds
    // how many rows (and the models they reference) are alive at once.
    const int Batch = 1024;
    var results = new Result[Batch];
    for (int start = 0; start < list.Count; start += Batch)
    {
        int n = Math.Min(Batch, list.Count - start);
        Parallel.For(0, n, par, i => results[i] = Export(list[start + i]));
        for (int i = 0; i < n; i++)
        {
            var r = results[i]; results[i] = default;
            ok += r.Ok; fail += r.Fail; embedded += r.Embedded; writeFail += r.WriteFail;
            if (r.DecodeFailure is { } reason)
            {
                int count = modelDecodeFailures.GetValueOrDefault(reason.Code) + 1;
                modelDecodeFailures[reason.Code] = count;
                if (count <= 3)
                    Console.Error.WriteLine($"  model decode failed at 0x{list[start + i].Offset:X}: {reason.Code} ({reason.Detail})");
            }
            if (r.Error != null && fail <= 5) Console.WriteLine(r.Error);
            if (r.Row != null) Row(r.Row);
        }
        Console.WriteLine($"  exported {start + n:N0}/{list.Count:N0}");
    }

    Result Export(ArchiveEntry e)
    {
        int ok = 0, fail = 0, embedded = 0;
        string stem = Util.Safe(Path.GetFileNameWithoutExtension(e.Name));
        try
        {
            var data = RecoveredResource.Read(src, e);
            int split = e.ResourceSystemBytes ?? checked((int)e.Segments![0].Size);
            var sys = data.AsSpan(0, split); var gfx = data.AsSpan(split);
            if (mode == "models")
            {
                var m = RageModelDecoder.Decode(sys, gfx, e.ResourceDrawableOffset);
                if (m == null)
                    return new Result(null, 0, 1, 0, ModelDecodeFailure.Explain(sys, gfx.Length), null, 0);
                string obj = Path.Combine(sub, stem + ".obj");
                string objText = RageModelDecoder.CreateObj(m);
                // A failed obj write means the model is missing from disk: fatal for the run (retryable),
                // unlike the decode failures above, which no rerun would change.
                try { if (!File.Exists(obj)) Util.WriteAllText(obj, objText); }
                catch (Exception wex) { return new Result(null, 0, 1, 0, null, $"  fail {e.Name}: {wex.GetType().Name} {wex.Message}", 1); }
                int wfail = 0;
                if (m.EmbeddedTextures != null && m.EmbeddedTextureDictionaryOffset is int dOff)
                {
                    // A directory or DDS write failure leaves the embedded texture missing from disk: retryable,
                    // so it fails the run like the obj write above. Embedded decode failures count like the
                    // textures branch counts dictionary decode failures: reported, non-fatal for the run.
                    string td;
                    try { td = Path.Combine(sub, stem + "_tex"); Directory.CreateDirectory(td); }
                    catch (Exception wex) { return new Result(null, 0, 1, 0, null, $"  fail {e.Name}: {wex.GetType().Name} {wex.Message}", 1); }
                    foreach (var t in m.EmbeddedTextures)
                    {
                        string p = Path.Combine(td, $"{t.Index:D3}_{Util.Safe(t.Name)}.dds");
                        if (File.Exists(p)) { embedded++; continue; }
                        byte[] dds;
                        try { dds = Util.BgraDds(RageTextureDecoder.DecodeEmbeddedTexturePixels(sys, gfx, dOff, t.Index)); }
                        catch { fail++; continue; }
                        try { Util.WriteAllBytes(p, dds); embedded++; }
                        catch (Exception wex) { fail++; wfail++; if (wfail <= 5) Console.WriteLine($"  fail {p}: {wex.GetType().Name} {wex.Message}"); }
                    }
                }
                return new Result(new { e.Offset, Drawable = e.ResourceDrawableOffset, File = stem + ".obj", m.Name, Meshes = m.Meshes.Count, Verts = m.Meshes.Sum(x => x.Positions.Count), Tris = m.Meshes.Sum(x => x.Indices.Count) / 3, m.IsFragment, m.HasBoneBindings,
                    Min = new[] { m.BoundsMin.X, m.BoundsMin.Y, m.BoundsMin.Z }, Max = new[] { m.BoundsMax.X, m.BoundsMax.Y, m.BoundsMax.Z },
                    // Per geometry, in the order of the OBJ's `o` groups: triangles, the model it belongs to in the
                    // drawable's primary-LOD list and its position in that model (a dehydrated BLAS leaf's geometry
                    // index counts the geometries whose shader it keeps, docs/bvh.md §4a).
                    MeshShaders = m.Meshes.Select(x => x.ShaderIndex).ToArray(),
                    MeshTris = m.Meshes.Select(x => x.Indices.Count / 3).ToArray(),
                    MeshModels = m.Meshes.Select(x => Util.MeshKey(x.Name).Model).ToArray(),
                    MeshGeoms = m.Meshes.Select(x => Util.MeshKey(x.Name).Geometry).ToArray(),
                    Embedded = m.EmbeddedTextures?.Select(t => new { t.Index, t.Name, t.Width, t.Height, t.Format, Hash = t.NameHash.ToString("X8"), Key = t.AssetKey?.ToString("X16") }).ToArray(),
                    Materials = m.Materials?.Select(x => new { x.ShaderIndex, Shader = x.ShaderHash.ToString("X8"), Preset = x.PresetHash.ToString("X8"), x.RenderBucket,
                        Textures = x.Textures.Select(r => new { Param = r.ParameterHash.ToString("X8"), r.TextureIndex, Name = r.NameHash.ToString("X8"), Id64 = r.OpaqueId64.ToString("X16"), Tag = r.Tag.ToString("X8") }).ToArray() }).ToArray() },
                    1, fail, embedded, null, null, wfail);
            }
            else
            {
                var dict = RageTextureDecoder.Inspect(sys, gfx);
                if (dict == null) return new Result(null, 0, 1, 0, null, null, 0);
                string td;
                try { td = Path.Combine(sub, stem); Directory.CreateDirectory(td); }
                catch (Exception wex) { return new Result(null, 0, 1, 0, null, $"  fail {e.Name}: {wex.GetType().Name} {wex.Message}", 1); }
                int wfail = 0;
                foreach (var t in dict.Textures)
                {
                    string p = Path.Combine(td, $"{t.Index:D3}_{Util.Safe(t.Name)}.dds");
                    if (File.Exists(p)) { ok++; continue; }
                    // A decoder problem is counted like today and no rerun would change it; a failed DDS write
                    // is retryable, so it additionally counts as a write failure and fails the run. A retryable
                    // read I/O error on the resource belongs to the same retryable class.
                    byte[] dds;
                    try { dds = RageTextureDecoder.CreateDds(sys, gfx, t.Index); }
                    catch (Exception ex) when (ex is IOException) { fail++; wfail++; if (wfail <= 5) Console.WriteLine($"  fail {p}: {ex.GetType().Name} {ex.Message}"); continue; }
                    catch { fail++; continue; }
                    try { Util.WriteAllBytes(p, dds); ok++; }
                    catch { fail++; wfail++; }
                }
                return new Result(new { e.Offset, Dir = stem, Textures = dict.Textures.Select(t => new { t.Index, t.Name, t.Width, t.Height, t.Format, t.MipCount, Hash = t.NameHash.ToString("X8"), Key = t.AssetKey?.ToString("X16") }).ToArray() },
                    ok, fail, 0, null, null, wfail);
            }
        }
        // A read I/O error while reading the resource is retryable like a write failure, so it also
        // withholds the manifest; structural exceptions stay non-retryable.
        catch (Exception ex) { return new Result(null, ok, fail + 1, embedded, null, $"  fail {e.Name}: {ex.GetType().Name} {ex.Message}", ex is IOException ? 1 : 0); }
    }
    if (mode == "models") Console.WriteLine($"  embedded textures {embedded:N0}");
    writer.WriteEndArray();
    if (mode == "models")
    {
        writer.WritePropertyName("ModelDecodeFailures");
        JsonSerializer.Serialize(writer, modelDecodeFailures, json);
        foreach (var (code, count) in modelDecodeFailures)
            Console.Error.WriteLine($"  model decode failures ({code}): {count:N0}");
    }
    writer.WriteEndObject();
    writer.Dispose();          // flushes; the later using-dispose is a no-op
    mstream.Dispose();
    // The manifest is the completion marker make_game resumes on. Like ttpf, keep the successful exports for
    // a retry, but a run with write failures must not mark the stage finished: withdraw a stale marker and
    // drop the partial instead of publishing it.
    if (writeFail == 0)
        File.Move(partial, manifestPath, true);
    else
    {
        Console.WriteLine($"  {writeFail} write failure(s): manifest withheld; rerun retries the missing files");
        try { WithdrawMarker(manifestPath); WithdrawMarker(partial); }
        catch (Exception ex) { Console.WriteLine($"  cannot withdraw the stale manifest: {ex.GetType().Name} {ex.Message}"); return 1; }
    }
}
Console.WriteLine($"{mode} {baseName}: ok={ok} fail={fail} skip={skip} in {sw.Elapsed.TotalSeconds:0}s");
return (mode == "ttpf" && fail != 0) || writeFail != 0 ? 1 : 0;

// A stale marker may be read-only from an earlier interrupted run; clear the attribute so the withdraw works.
static void WithdrawMarker(string path)
{
    if (!File.Exists(path)) return;
    try { File.SetAttributes(path, FileAttributes.Normal); } catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
    File.Delete(path);
}

static void Usage()
{
    Console.WriteLine("usage: rpf9dump <mode> <source> <outdir> [--jobs N] [--start BYTES] [--length BYTES]");
    Console.WriteLine("  ttpf      texture dictionaries and world texture pools as DDS");
    Console.WriteLine("  models    drawable models as OBJ, with their embedded textures");
    Console.WriteLine("  textures  texture dictionaries as DDS");
    Console.WriteLine("  audio     AWC banks whole, plus audio_manifest.json");
    Console.WriteLine("  xma       XMA2 streams as RIFF XMA2 .wav for ffmpeg");
}

// One exported entry: its manifest row (null when nothing was written), counts, a failure to report in order,
// and how many of its failures were output-write failures (the ones a retry can fix; decode failures are not).
readonly record struct Result(object? Row, int Ok, int Fail, int Embedded, ModelDecodeFailure? DecodeFailure, string? Error, int WriteFail);
