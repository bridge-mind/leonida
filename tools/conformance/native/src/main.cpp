#include <windows.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

#include "diagnostics.h"
#include "gpu.h"
#include "manifest.h"
#include "png_io.h"
#include "renderer.h"
#include "scene_plan.h"
#include "scenes.h"
#include "verify.h"

namespace {

// Balances CoInitializeEx for the lifetime of main. One owner, no copy: a second
// initialisation would change the apartment model under WIC.
struct ComScope {
    HRESULT result = E_FAIL;

    ComScope() { result = CoInitializeEx(nullptr, COINIT_MULTITHREADED); }
    ~ComScope() {
        if (SUCCEEDED(result)) {
            CoUninitialize();
        }
    }
    ComScope(const ComScope&) = delete;
    ComScope& operator=(const ComScope&) = delete;
};

struct CommandLineOptions {
    std::filesystem::path output_directory;
    std::filesystem::path shader_directory;  // empty: <executable directory>\shaders
    int adapter_index = -1;                  // -1: the non-software adapter with most memory
    bool enable_debug_layer = false;
    bool show_help = false;
};

// Totals for the run. Readback and PNG are counted separately so a failure says which
// path broke: rendering, or the file the diff tool will read.
struct RunTally {
    int readback_passed = 0;
    int readback_failed = 0;
    int png_passed = 0;
    int png_failed = 0;
};

void PrintUsage() {
    std::printf(
        "usage: conform_native <output-directory> [options]\n"
        "  --shaders <directory>  where the .cso files live (default: <exe>\\shaders)\n"
        "  --adapter <index>      adapter from the printed list (default: most dedicated\n"
        "                         memory, non-software)\n"
        "  --debug-layer          enable D3D12 validation and fail on validation errors\n"
        "  --help                 print this text\n");
}

// Parses argv into options. Returns false on any malformed argument; the caller prints
// usage, so this function prints only the reason.
bool ParseCommandLine(int argument_count, char** arguments, CommandLineOptions& options) {
    for (int index = 1; index < argument_count; ++index) {
        const std::string argument = arguments[index];
        if (argument == "--help" || argument == "-h") {
            options.show_help = true;
            continue;
        }
        if (argument == "--debug-layer") {
            options.enable_debug_layer = true;
            continue;
        }
        if (argument == "--adapter" || argument == "--shaders") {
            if (index + 1 >= argument_count) {
                ReportFailure(__FILE__, __LINE__, "ParseCommandLine",
                              argument + " needs a value", "see --help for the argument list");
                return false;
            }
            const std::string value = arguments[++index];
            if (argument == "--shaders") {
                options.shader_directory = std::filesystem::path(value);
                continue;
            }
            char* end_of_number = nullptr;
            const long parsed = std::strtol(value.c_str(), &end_of_number, 10);
            if (end_of_number == value.c_str() || *end_of_number != '\0' || parsed < 0) {
                ReportFailure(__FILE__, __LINE__, "ParseCommandLine",
                              "--adapter expects a non-negative index, got \"" + value + "\"",
                              "the adapter list printed at startup numbers the adapters");
                return false;
            }
            options.adapter_index = static_cast<int>(parsed);
            continue;
        }
        if (!argument.empty() && argument[0] == '-') {
            ReportFailure(__FILE__, __LINE__, "ParseCommandLine", "unknown option " + argument,
                          "see --help for the argument list");
            return false;
        }
        if (!options.output_directory.empty()) {
            ReportFailure(__FILE__, __LINE__, "ParseCommandLine",
                          "more than one output directory: " + argument,
                          "pass exactly one output directory");
            return false;
        }
        options.output_directory = std::filesystem::path(argument);
    }

    if (options.show_help) {
        return true;
    }
    if (options.output_directory.empty()) {
        ReportFailure(__FILE__, __LINE__, "ParseCommandLine", "no output directory given",
                      "pass the directory the PNGs and manifest.json should be written to");
        return false;
    }
    return true;
}

// The directory holding the running executable. Shaders are copied next to the exe at
// build time, so this keeps the tool runnable from any working directory.
bool ExecutableDirectory(std::filesystem::path& directory_out) {
    std::wstring buffer(32768, L'\0');
    const DWORD length =
        GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length == 0 || length >= buffer.size()) {
        ReportFailure(__FILE__, __LINE__, "GetModuleFileNameW",
                      "could not read the executable path",
                      "the process image is still on disk and the path is shorter than 32768 "
                      "characters");
        return false;
    }
    buffer.resize(length);
    directory_out = std::filesystem::path(buffer).parent_path();
    return true;
}

// Trims the trailing period Windows appends to system messages so the "Check:" clause
// that follows does not produce "...exists.. Check:".
std::string TrimTrailingPeriod(std::string text) {
    while (!text.empty() && text.back() == '.') {
        text.pop_back();
    }
    return text;
}

// Creates the output directory if needed. Reports the filesystem's own message, which
// distinguishes "it is a file" from "access denied" from "the drive is gone".
bool EnsureOutputDirectory(const std::filesystem::path& directory) {
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error) {
        ReportFailure(__FILE__, __LINE__, "std::filesystem::create_directories",
                      TrimTrailingPeriod("could not create " + directory.string() + ": " +
                                         error.message()),
                      "the path is a directory the process can write to, not an existing file and "
                      "not a read-only volume");
        return false;
    }
    return true;
}

// Renders, verifies and writes one scene. Returns false only when the pipeline itself
// broke; a wrong pixel is recorded in the tally and the run keeps going so one log
// shows every failing scene.
bool RunOneScene(GpuContext& gpu, Renderer& renderer,
                 const std::filesystem::path& output_directory, int scene_index,
                 const conform::Scene& scene, RunTally& tally) {
    const DrawPlan plan = MakeDrawPlan(scene, scene_index);
    PixelSurface rendered;
    if (!renderer.RenderScene(gpu, plan, rendered)) {
        ReportFailure(__FILE__, __LINE__, "Renderer::RenderScene",
                      std::string("scene ") + scene.name + " did not render",
                      "the failure printed above this line names the call that broke");
        return false;
    }
    if (rendered.width != conform::kSceneWidth || rendered.height != conform::kSceneHeight) {
        ReportFailure(__FILE__, __LINE__, "Renderer::RenderScene",
                      std::string("scene ") + scene.name + " produced a " +
                          std::to_string(rendered.width) + "x" + std::to_string(rendered.height) +
                          " image",
                      "the readback footprint matches the contract's 256x256 target");
        return false;
    }

    std::printf("\nscene %s: %d checks -- %s\n", scene.name, scene.checkCount, scene.proves);
    const CheckTally readback_tally = VerifyChecks(scene, rendered, "readback");
    tally.readback_passed += readback_tally.passed;
    tally.readback_failed += readback_tally.failed;

    const std::filesystem::path png_path =
        output_directory / (std::string(scene.name) + ".png");
    if (!WritePngFile(png_path, rendered)) {
        return false;
    }

    // Decode what landed on disk. A WIC channel swap or a truncated file passes the
    // readback checks and fails here, which is the point of the second comparison.
    PixelSurface decoded;
    if (!ReadPngFile(png_path, decoded)) {
        return false;
    }
    if (decoded.width != rendered.width || decoded.height != rendered.height) {
        ReportFailure(__FILE__, __LINE__, "ReadPngFile",
                      png_path.string() + " decoded to " + std::to_string(decoded.width) + "x" +
                          std::to_string(decoded.height),
                      "the encoder wrote a full 256x256 image");
        return false;
    }
    const CheckTally png_tally = VerifyChecks(scene, decoded, "png");
    tally.png_passed += png_tally.passed;
    tally.png_failed += png_tally.failed;
    return true;
}

// Every scene in the contract, in declaration order, then the manifest. Scene order is
// fixed so two runs do the same GPU work in the same order.
bool RunAllScenes(GpuContext& gpu, Renderer& renderer,
                  const std::filesystem::path& output_directory, RunTally& tally) {
    for (int index = 0; index < conform::kSceneCount; ++index) {
        if (!RunOneScene(gpu, renderer, output_directory, index, conform::kScenes[index], tally)) {
            return false;
        }
    }
    // After the images: a manifest must never describe PNGs that do not exist.
    return WriteManifest(output_directory / "manifest.json", gpu.adapter_description,
                         conform::kScenes, conform::kSceneCount);
}

}  // namespace

int main(int argument_count, char** arguments) {
    // stdout unbuffered so PASS/FAIL lines stay in order with FAIL diagnostics, which
    // stderr prints immediately. A reordered transcript invites a wrong conclusion.
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    CommandLineOptions options;
    if (!ParseCommandLine(argument_count, arguments, options)) {
        PrintUsage();
        return 2;
    }
    if (options.show_help) {
        PrintUsage();
        return 0;
    }

    std::printf("conform native renderer (D3D12, 256x256 R8G8B8A8_UNORM, offscreen)\n");
    ComScope com;
    if (FAILED(com.result)) {
        HrSucceeded(com.result, __FILE__, __LINE__, "CoInitializeEx",
                    "COM can be initialised on this thread; RPC_E_CHANGED_MODE means another "
                    "part of the process already picked a different apartment");
        return 1;
    }

    std::filesystem::path shader_directory = options.shader_directory;
    if (shader_directory.empty()) {
        std::filesystem::path executable_directory;
        if (!ExecutableDirectory(executable_directory)) {
            return 1;
        }
        shader_directory = executable_directory / "shaders";
    }
    if (!EnsureOutputDirectory(options.output_directory)) {
        return 1;
    }
    std::printf("output:  %s\n", options.output_directory.string().c_str());
    std::printf("shaders: %s\n", shader_directory.string().c_str());

    GpuContext gpu;
    if (!CreateGpuContext(gpu, options.adapter_index, options.enable_debug_layer)) {
        return 1;
    }
    std::printf("adapter: %s%s\n\n", gpu.adapter_description.c_str(),
                gpu.adapter_is_software ? " (software)" : "");

    // A run that asked for validation and did not get it must stop here. Reporting
    // "0 validation errors" after the layer failed to load is an invented result.
    if (options.enable_debug_layer && !gpu.validation_active) {
        std::fprintf(stderr,
                     "validation: NOT RUN - --debug-layer was requested but the D3D12 debug "
                     "layer is not installed on this machine. Install the Graphics Tools "
                     "optional feature, or drop --debug-layer to run unvalidated.\n");
        return 1;
    }

    Renderer renderer;
    if (!renderer.Initialize(gpu, shader_directory)) {
        return 1;
    }

    RunTally tally;
    const bool run_completed = RunAllScenes(gpu, renderer, options.output_directory, tally);

    bool validation_errors_present = false;
    if (options.enable_debug_layer) {
        const int validation_error_count = ReportValidationMessages(*gpu.device.Get());
        validation_errors_present = validation_error_count > 0;
        std::printf("\nvalidation errors: %d\n", validation_error_count);
    }

    std::printf("\nreadback checks: %d passed, %d failed\n", tally.readback_passed,
                tally.readback_failed);
    std::printf("png checks:      %d passed, %d failed\n", tally.png_passed, tally.png_failed);

    if (!run_completed) {
        std::printf("RESULT: FAIL (run did not complete)\n");
        return 1;
    }
    const bool all_checks_passed =
        tally.readback_failed == 0 && tally.png_failed == 0 && !validation_errors_present;
    std::printf("RESULT: %s\n", all_checks_passed ? "PASS" : "FAIL");
    return all_checks_passed ? 0 : 1;
}
