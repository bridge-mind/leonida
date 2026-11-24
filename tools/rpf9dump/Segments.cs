// Reading a recovered resource from its own compression blocks, without an ArchiveDocument.
//
// The recovery step (Rpf9ModelRecovery/Rpf9TextureRecovery) records, for every entry it recovers, the file offsets
// of the zlib blocks that make up its system and graphics segments. ArchiveReader.ReadResourceData reaches the same
// bytes through an ArchiveDocument, which needs a recognizable archive header. A --start/--length window is a raw
// slice of an archive, so the header may be at an offset the slice does not contain (a window of levels.rpf) or
// absent entirely; these entries are then read from their blocks alone.
using System.IO.Compression;
using RpfPreview.Core;

/// <summary>Reads a recovered resource that has no archive document behind it.</summary>
public static class RecoveredResource
{
    /// <summary>The limit ArchiveReader puts on a recovered resource; the decoder is not given more than this.</summary>
    public const int MaxBytes = 256 * 1024 * 1024;

    /// <summary>The resource's inflated bytes, block by block, in stored order.</summary>
    public static byte[] Read(string path, ArchiveEntry entry)
    {
        if (entry.Segments is not { Count: > 0 } segments || entry.Size < 1 || entry.Size > MaxBytes)
            throw new InvalidDataException("The recovered resource has no blocks or exceeds the 256 MiB limit.");
        byte[] data = new byte[checked((int)entry.Size)];
        using var file = File.Open(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        int written = 0;
        foreach (var block in segments)
        {
            if (block.Offset < 0 || block.StoredSize < 6 || block.Size < 0 || block.Offset + block.StoredSize > file.Length)
                throw new InvalidDataException("A recovered block runs past the file.");
            if (block.Size > data.Length - written)
                throw new InvalidDataException("The recovered blocks exceed the resource size.");
            file.Position = block.Offset;
            using Stream stored = new StoredBlock(file, block.StoredSize);
            using Stream content = block.IsRaw ? stored : new ZLibStream(stored, CompressionMode.Decompress);
            content.ReadExactly(data.AsSpan(written, checked((int)block.Size)));
            written += (int)block.Size;
        }
        if (written != data.Length) throw new InvalidDataException("The recovered resource is truncated.");
        return data;
    }

    /// <summary>A read-only view of one stored block, so a compressed block is never copied whole before inflating.</summary>
    private sealed class StoredBlock(Stream inner, long length) : Stream
    {
        private long remaining = length;
        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
        public override int Read(byte[] buffer, int offset, int count)
        {
            if (remaining <= 0) return 0;
            int read = inner.Read(buffer, offset, (int)Math.Min(count, remaining));
            if (read == 0) throw new InvalidDataException("A stored block is truncated.");
            remaining -= read;
            return read;
        }
        public override void Flush() { }
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        // Deliberately no Dispose: the caller owns the file and the next block reads from it.
    }
}
