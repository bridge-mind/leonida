import sys
import os
import unittest

sys.path.insert(0, 'tools')
import rdna2spirv
import spirv_asm

class MeshAllocSingleWriterTests(unittest.TestCase):
    def test_ms_alloc_req_single_writer_guard(self):
        """SHADER-766: s_sendmsg(MSG_GS_ALLOC_REQ) restricts OpStore %ms_counts to local_index == 0."""
        for mesh_compute in (False, True):
            orig_mc = rdna2spirv.MESH_COMPUTE
            try:
                rdna2spirv.MESH_COMPUTE = mesh_compute
                gen = rdna2spirv.Gen("ms")
                gen.ms = {"threads": 128, "maxv": 64, "maxp": 64, "clip": 0, "cull": 0, "params": []}
                # Emit MSG_GS_ALLOC_REQ
                ins = {
                    "op": "s_sendmsg",
                    "args": [{"k": "special", "t": "sendmsg(MSG_GS_ALLOC_REQ)"}],
                    "mods": []
                }
                gen._insn(ins)

                body_text = "\n".join(gen.body)
                # 1. OpStore %ms_counts must be present
                self.assertIn("OpStore %ms_counts", body_text)

                # 2. Must load %local_index and compare to 0
                self.assertIn("= OpLoad %uint %local_index", body_text)
                self.assertRegex(body_text, r"%(\w+) = OpIEqual %bool %(\w+) %c0")

                # 3. Must have structured selection merge and conditional branch
                self.assertRegex(body_text, r"OpSelectionMerge %(\w+) None\s+OpBranchConditional %(\w+) %(\w+) %(\w+)")

                # 4. OpStore %ms_counts must only appear inside the conditional block
                lines = [l.strip() for l in gen.body if l.strip()]
                store_idx = next(i for i, l in enumerate(lines) if "OpStore %ms_counts" in l)
                branch_cond_idx = next(i for i, l in enumerate(lines) if "OpBranchConditional" in l)
                self.assertGreater(store_idx, branch_cond_idx)

                # 5. Verify assembly into a complete module passes spirv_asm
                # Build mock header
                types_and_vars = [
                    "OpCapability Shader",
                    "OpCapability MeshShadingEXT" if not mesh_compute else "",
                    "OpMemoryModel Logical GLSL450",
                    'OpEntryPoint GLCompute %main "main" %local_index' if mesh_compute else 'OpEntryPoint MeshEXT %main "main" %local_index',
                    "OpExecutionMode %main LocalSize 128 1 1",
                    "OpDecorate %local_index BuiltIn LocalInvocationIndex",
                    "%void = OpTypeVoid",
                    "%bool = OpTypeBool",
                    "%uint = OpTypeInt 32 0",
                    "%c0 = OpConstant %uint 0",
                    "%_ptr_Input_uint = OpTypePointer Input %uint",
                    "%_ptr_Workgroup_uint = OpTypePointer Workgroup %uint",
                    "%_ptr_Function_uint = OpTypePointer Function %uint",
                    "%local_index = OpVariable %_ptr_Input_uint Input",
                    "%ms_counts = OpVariable %_ptr_Workgroup_uint Workgroup",
                    "%m0 = OpVariable %_ptr_Function_uint Function",
                    "%fn_t = OpTypeFunction %void",
                    "%main = OpFunction %void None %fn_t",
                    "%entry = OpLabel"
                ]
                spv_text = "\n".join([line for line in types_and_vars if line]) + "\n" + body_text + "\nOpReturn\nOpFunctionEnd\n"
                bin_data = spirv_asm.assemble(spv_text)
                self.assertGreater(len(bin_data), 0)
            finally:
                rdna2spirv.MESH_COMPUTE = orig_mc

if __name__ == '__main__':
    unittest.main()
