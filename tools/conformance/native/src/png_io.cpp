#include "png_io.h"

#include <wincodec.h>
#include <wrl/client.h>

#include <string>
#include <vector>

#include "diagnostics.h"

using Microsoft::WRL::ComPtr;

namespace {

// One WIC factory per call. It is a few thousand bytes of COM state, and keeping it
// local means no object outlives the function that needed it.
bool CreateWicFactory(ComPtr<IWICImagingFactory>& factory_out) {
    return HrSucceeded(
        CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                         IID_PPV_ARGS(factory_out.GetAddressOf())),
        __FILE__, __LINE__, "CoCreateInstance(CLSID_WICImagingFactory)",
        "the Windows Imaging Component is registered; a stripped Windows image without "
        "the WIC feature is the usual cause");
}

// WIC's PNG encoder takes 32bppBGRA. Swapping here, once, keeps the channel order
// correct on disk without a format converter.
std::vector<uint8_t> ConvertRgbaToBgra(const PixelSurface& surface) {
    std::vector<uint8_t> bgra(surface.rgba.size());
    for (size_t offset = 0; offset + 3 < surface.rgba.size(); offset += 4) {
        bgra[offset + 0] = surface.rgba[offset + 2];
        bgra[offset + 1] = surface.rgba[offset + 1];
        bgra[offset + 2] = surface.rgba[offset + 0];
        bgra[offset + 3] = surface.rgba[offset + 3];
    }
    return bgra;
}

}  // namespace

bool WritePngFile(const std::filesystem::path& path, const PixelSurface& surface) {
    ComPtr<IWICImagingFactory> factory;
    if (!CreateWicFactory(factory)) {
        return false;
    }
    if (surface.width <= 0 || surface.height <= 0 ||
        surface.rgba.size() != static_cast<size_t>(surface.width) * surface.height * 4) {
        ReportFailure(__FILE__, __LINE__, "WritePngFile",
                      "the pixel surface is empty or its size does not match its dimensions",
                      "the readback path produced a full 256x256 image before this ran");
        return false;
    }

    ComPtr<IWICStream> stream;
    if (!HrSucceeded(factory->CreateStream(stream.GetAddressOf()), __FILE__, __LINE__,
                     "IWICImagingFactory::CreateStream", "WIC is initialised on this thread")) {
        return false;
    }
    const std::string write_hint =
        "the output directory is writable and nothing else holds " + path.string() + " open";
    if (!HrSucceeded(stream->InitializeFromFilename(path.c_str(), GENERIC_WRITE), __FILE__, __LINE__,
                     "IWICStream::InitializeFromFilename", write_hint.c_str())) {
        return false;
    }

    ComPtr<IWICBitmapEncoder> encoder;
    if (!HrSucceeded(factory->CreateEncoder(GUID_ContainerFormatPng, nullptr,
                                            encoder.GetAddressOf()),
                     __FILE__, __LINE__, "IWICImagingFactory::CreateEncoder(PNG)",
                     "the PNG codec is installed with Windows")) {
        return false;
    }
    if (!HrSucceeded(encoder->Initialize(stream.Get(), WICBitmapEncoderNoCache), __FILE__, __LINE__,
                     "IWICBitmapEncoder::Initialize", "the output stream is writable")) {
        return false;
    }

    ComPtr<IWICBitmapFrameEncode> frame;
    ComPtr<IPropertyBag2> frame_properties;
    if (!HrSucceeded(encoder->CreateNewFrame(frame.GetAddressOf(), frame_properties.GetAddressOf()),
                     __FILE__, __LINE__, "IWICBitmapEncoder::CreateNewFrame",
                     "the encoder was initialised")) {
        return false;
    }
    if (!HrSucceeded(frame->Initialize(frame_properties.Get()), __FILE__, __LINE__,
                     "IWICBitmapFrameEncode::Initialize", "the frame property bag is valid")) {
        return false;
    }
    if (!HrSucceeded(frame->SetSize(static_cast<UINT>(surface.width),
                                    static_cast<UINT>(surface.height)),
                     __FILE__, __LINE__, "IWICBitmapFrameEncode::SetSize",
                     "the dimensions are non-zero and fit in UINT")) {
        return false;
    }

    WICPixelFormatGUID pixel_format = GUID_WICPixelFormat32bppBGRA;
    if (!HrSucceeded(frame->SetPixelFormat(&pixel_format), __FILE__, __LINE__,
                     "IWICBitmapFrameEncode::SetPixelFormat", "the PNG codec accepts 32bppBGRA")) {
        return false;
    }
    if (!IsEqualGUID(pixel_format, GUID_WICPixelFormat32bppBGRA)) {
        ReportFailure(__FILE__, __LINE__, "IWICBitmapFrameEncode::SetPixelFormat",
                      "the PNG codec silently changed the requested pixel format",
                      "a channel order assumption in this tool no longer holds; the written "
                      "file would not be comparable");
        return false;
    }

    std::vector<uint8_t> bgra = ConvertRgbaToBgra(surface);
    const UINT stride = static_cast<UINT>(surface.width) * 4;
    if (!HrSucceeded(frame->WritePixels(static_cast<UINT>(surface.height), stride,
                                        static_cast<UINT>(bgra.size()), bgra.data()),
                     __FILE__, __LINE__, "IWICBitmapFrameEncode::WritePixels",
                     "the disk has space and the surface buffer is as large as its dimensions")) {
        return false;
    }
    if (!HrSucceeded(frame->Commit(), __FILE__, __LINE__, "IWICBitmapFrameEncode::Commit",
                     "the frame was fully written before commit")) {
        return false;
    }
    // Commit is where the PNG chunks are flushed; the stream closes when the local COM
    // pointers are released as this function returns.
    return HrSucceeded(encoder->Commit(), __FILE__, __LINE__, "IWICBitmapEncoder::Commit",
                       "the stream accepted every byte; a full disk shows up here");
}

bool ReadPngFile(const std::filesystem::path& path, PixelSurface& surface_out) {
    ComPtr<IWICImagingFactory> factory;
    if (!CreateWicFactory(factory)) {
        return false;
    }

    ComPtr<IWICBitmapDecoder> decoder;
    const std::string read_hint = path.string() +
                                  " exists and is not locked by another process";
    if (!HrSucceeded(factory->CreateDecoderFromFilename(path.c_str(), nullptr, GENERIC_READ,
                                                        WICDecodeMetadataCacheOnDemand,
                                                        decoder.GetAddressOf()),
                     __FILE__, __LINE__, "IWICImagingFactory::CreateDecoderFromFilename",
                     read_hint.c_str())) {
        return false;
    }

    ComPtr<IWICBitmapFrameDecode> frame;
    if (!HrSucceeded(decoder->GetFrame(0, frame.GetAddressOf()), __FILE__, __LINE__,
                     "IWICBitmapDecoder::GetFrame(0)", "the PNG has at least one frame")) {
        return false;
    }

    ComPtr<IWICFormatConverter> converter;
    if (!HrSucceeded(factory->CreateFormatConverter(converter.GetAddressOf()), __FILE__, __LINE__,
                     "IWICImagingFactory::CreateFormatConverter", "WIC is initialised")) {
        return false;
    }
    if (!HrSucceeded(converter->Initialize(frame.Get(), GUID_WICPixelFormat32bppRGBA,
                                           WICBitmapDitherTypeNone, nullptr, 0.0,
                                           WICBitmapPaletteTypeCustom),
                     __FILE__, __LINE__, "IWICFormatConverter::Initialize",
                     "the decoded frame can be converted to 32bppRGBA without a palette")) {
        return false;
    }

    UINT width = 0;
    UINT height = 0;
    if (!HrSucceeded(converter->GetSize(&width, &height), __FILE__, __LINE__,
                     "IWICFormatConverter::GetSize", "the frame header is intact")) {
        return false;
    }
    surface_out.width = static_cast<int>(width);
    surface_out.height = static_cast<int>(height);
    surface_out.rgba.resize(static_cast<size_t>(width) * height * 4);
    return HrSucceeded(converter->CopyPixels(nullptr, width * 4,
                                             static_cast<UINT>(surface_out.rgba.size()),
                                             surface_out.rgba.data()),
                       __FILE__, __LINE__, "IWICFormatConverter::CopyPixels",
                       "the file holds a complete image; a truncated PNG fails here");
}
