using RpfPreview.Core;

internal readonly record struct ModelDecodeFailure(string Code, string Detail)
{
    // The decoder returns null for all rejected layouts. Probe the resource
    // again only on failure, so successful exports keep the existing path.
    public static ModelDecodeFailure Explain(ReadOnlySpan<byte> system, int graphicsBytes)
    {
        if (system.Length > RageModelDecoder.MaxSystemBytes)
            return new("system-size-limit", $"{system.Length} system bytes");
        if (graphicsBytes > RageModelDecoder.MaxGraphicsBytes)
            return new("graphics-size-limit", $"{graphicsBytes} graphics bytes");

        var extent = RageModelDecoder.ProbeRequiredExtents(system);
        if (extent is null)
            return RageModelDecoder.LooksLikeResource(system)
                ? new("invalid-or-unsupported-layout", "recognized header, resource metadata probe failed")
                : new("unrecognized-header", $"{system.Length} system bytes");
        if (extent.RequiredSystemBytes > system.Length)
            return new("truncated-system", $"{system.Length}/{extent.RequiredSystemBytes} system bytes");
        if (!extent.MetadataComplete)
            return new("incomplete-metadata", "resource metadata probe did not complete");
        if (extent.RequiredGraphicsBytes > graphicsBytes)
            return new("truncated-graphics", $"{graphicsBytes}/{extent.RequiredGraphicsBytes} graphics bytes");
        if (graphicsBytes == 0)
            return new("missing-graphics", "empty graphics segment");
        return new("decode-rejected", "size and metadata checks passed");
    }
}
