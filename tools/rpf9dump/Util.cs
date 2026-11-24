// Helpers shared by rpf9dump's exporters (linked into tests/Rpf9DumpDiagnostics too).
using RpfPreview.Core;

public static class Util
{
    // Every output is written to a unique <path>.*.partial and renamed into place. The exporters skip any file that
    // already exists, and make_game.py takes manifest.json / audio_manifest.json as proof a step finished, so a
    // file cut short by a stopped run or a full disk must never appear under its final name.
    // Unique names matter on parallel export (two models with the same stem) and when a stale exclusive
    // lock on the classic ".partial" name would throw IOException: "being used by another process".
    public static void WriteAtomic(string path, Action<string> write)
    {
        TryDelete(path + ".partial");   // leftover from older builds / stopped runs
        string tmp = PartialName(path);
        try
        {
            write(tmp);
            MoveReplace(tmp, path);
        }
        finally { TryDelete(tmp); }
    }
    public static void WriteAllBytes(string path, byte[] data) => WriteAtomic(path, t => File.WriteAllBytes(t, data));
    public static void WriteAllText(string path, string text) => WriteAtomic(path, t => File.WriteAllText(t, text));

    public static string PartialName(string path) =>
        path + $".{Environment.ProcessId}.{Environment.CurrentManagedThreadId}.{Guid.NewGuid():N}.partial";

    public static void TryDelete(string path)
    {
        try { if (File.Exists(path)) File.Delete(path); }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }

    /// <summary>Create/truncate for exclusive write, retrying through brief file locks.</summary>
    public static FileStream OpenExclusiveCreate(string path, int bufferSize = 1 << 20)
    {
        IOException? last = null;
        for (int i = 0; i < 20; i++)
        {
            try
            {
                return new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, bufferSize);
            }
            catch (IOException ex)
            {
                last = ex;
                TryDelete(path);
                Thread.Sleep(50 * (i + 1));
            }
        }
        throw last!;
    }

    /// <summary>Atomic publish with retries (File.Move onto a briefly locked target can fail).</summary>
    public static void MoveReplace(string src, string dst)
    {
        IOException? last = null;
        for (int i = 0; i < 20; i++)
        {
            try
            {
                File.Move(src, dst, overwrite: true);
                return;
            }
            catch (IOException ex)
            {
                last = ex;
                Thread.Sleep(50 * (i + 1));
            }
        }
        throw last!;
    }

    // The decoder names every geometry mesh `model_{mi}_geometry_{gi}`, where mi is the position of the model in the
    // drawable's primary-LOD model list and gi the position of the geometry in that model - the pair a dehydrated
    // BLAS leaf stores as its geometry index (docs/bvh.md §4a). Other spellings give -1.
    public static (int Model, int Geometry) MeshKey(string name)
    {
        string[] parts = name.Split('_');
        if (parts.Length == 4 && parts[0] == "model" && parts[2] == "geometry" &&
            int.TryParse(parts[1], System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture, out int model) &&
            int.TryParse(parts[3], System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture, out int geometry))
            return (model, geometry);
        return (-1, -1);
    }

    public static string Safe(string s)
    {
        var bad = Path.GetInvalidFileNameChars();
        var c = s.Select(ch => bad.Contains(ch) || char.IsControl(ch) ? '_' : ch).ToArray();
        var v = new string(c).Trim(' ', '.'); return v.Length == 0 ? "x" : v[..Math.Min(120, v.Length)];
    }
    // Uncompressed 32-bit BGRA DDS (legacy header, A8R8G8B8 masks).
    public static byte[] BgraDds(RageTexturePixels px)
    {
        var o = new byte[128 + px.Bgra32.Length];
        void U(int at, uint v) => BitConverter.TryWriteBytes(o.AsSpan(at), v);
        U(0, 0x20534444); U(4, 124); U(8, 0x100F); U(12, (uint)px.Height); U(16, (uint)px.Width); U(20, (uint)(px.Width * 4));
        U(76, 32); U(80, 0x41); U(88, 32); U(92, 0x00FF0000); U(96, 0x0000FF00); U(100, 0x000000FF); U(104, 0xFF000000); U(108, 0x1000);
        px.Bgra32.CopyTo(o, 128); return o;
    }
}
