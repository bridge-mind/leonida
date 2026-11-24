"""Optional hardware oracle: run small hand-written AMD GPU programs natively and read the result back.

asm.py       program text -> GFX10/GFX11 kernels, HSA metadata, the SALU opcodes that differ (no hardware needed)
runner.py    finds AMD clang, the HIP runtime DLL and a host C compiler, builds hw_oracle.c, runs kernels
tsharp.py    image T# words per family (gfx10, gfx11) and the raw kernel that image_loads a texture (tools/image_fuzz.py)
hw_oracle.c  the launcher: loads a code object through the driver's HIP runtime and dumps each kernel's output words

Everything here is optional. When a tool is missing, runner.Unavailable is raised and callers exit 77.
See docs/shader_validation.md ("Hardware oracle").
"""
