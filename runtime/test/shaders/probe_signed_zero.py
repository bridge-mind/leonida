"""Run the SHADER-605 direct OpImageWrite probe with the supplied vk_store_test binary."""

import argparse
from pathlib import Path
import subprocess
import tempfile


# Hand-written compute shader. Only the image format changes between runs.
MODULE = """OpCapability Shader
OpCapability StorageImageWriteWithoutFormat
OpCapability SignedZeroInfNanPreserve
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %image
OpExecutionMode %main LocalSize 1 1 1
OpExecutionMode %main SignedZeroInfNanPreserve 32
OpDecorate %image DescriptorSet 0
OpDecorate %image Binding 8
%void = OpTypeVoid
%fn = OpTypeFunction %void
%i = OpTypeInt 32 1
%f = OpTypeFloat 32
%xy = OpTypeVector %i 2
%rgba = OpTypeVector %f 4
%img = OpTypeImage %f 2D 0 0 0 2 {image_format}
%ptr = OpTypePointer UniformConstant %img
%image = OpVariable %ptr UniformConstant
%zero = OpConstant %i 0
%one = OpConstant %i 1
%two = OpConstant %i 2
%a = OpConstantComposite %xy %zero %zero
%b = OpConstantComposite %xy %one %zero
%c = OpConstantComposite %xy %two %zero
%n = OpConstant %f -0.0
%p = OpConstant %f 0.0
%f1 = OpConstant %f 1.0
%va = OpConstantComposite %rgba %n %p %p %p
%vb = OpConstantComposite %rgba %n %f1 %p %p
%vc = OpConstantComposite %rgba %n %n %p %p
%main = OpFunction %void None %fn
%entry = OpLabel
%handle = OpLoad %img %image
OpImageWrite %handle %a %va
OpImageWrite %handle %b %vb
OpImageWrite %handle %c %vc
OpReturn
OpFunctionEnd
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("harness", type=Path)
    args = parser.parse_args()
    results = []
    with tempfile.TemporaryDirectory() as scratch:
        for image_format in ("Unknown", "Rgba32f"):
            print(image_format, flush=True)
            source = Path(scratch) / f"{image_format}.spvasm"
            binary = source.with_suffix(".spv")
            source.write_text(MODULE.format(image_format=image_format), encoding="utf-8", newline="\n")
            subprocess.run(["spirv-as", "--target-env", "vulkan1.2", source, "-o", binary], check=True)
            subprocess.run(["spirv-val", "--target-env", "vulkan1.2", binary], check=True)
            results.append(subprocess.run([args.harness.resolve(), binary, "signed-zero"]).returncode)
    if any(code not in (0, 77) for code in results):
        return 1
    return 77 if 77 in results else 0


if __name__ == "__main__":
    raise SystemExit(main())
