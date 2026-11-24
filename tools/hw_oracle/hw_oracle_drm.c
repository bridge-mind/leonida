/* Hardware oracle launcher for Linux. Runs kernels as raw compute dispatches through the kernel driver
 * (libdrm_amdgpu): no ROCm, no Vulkan, nothing between the instructions and the GPU.
 *
 *   hw_oracle_drm <code.bin> <words> <render node>
 *
 * <code.bin> is the .text section of the assembled module. Kernel tokens come on stdin, whitespace separated:
 *   name@offset/rsrc1/rsrc2/wave[:threads[:words]]
 *     offset        where the kernel starts in code.bin, a multiple of 256
 *     rsrc1, rsrc2  COMPUTE_PGM_RSRC1 and COMPUTE_PGM_RSRC2 from the kernel's descriptor: its register counts and
 *                   float mode, and its user and system SGPRs. drm.py reads them out of the assembled object and
 *                   puts the descriptor's group segment into RSRC2's LDS_SIZE field, which the assembler leaves 0
 *     wave          32 or 64, from the descriptor too
 * Each kernel gets what the Windows launcher (hw_oracle.c) gives it: s[0:1] points at an 8-byte kernarg holding
 * the address of a <words> dword buffer pre-filled with 0xDEAD0000 (1..16384 words), and it runs as one workgroup
 * of `threads` threads (1..1024), started the way its descriptor says.
 * Output, one line per kernel:  <name> <hex word> ...   The first line is "# device=<name> arch=<gfxNNNN> ...".
 * Exit 0 ok, 77 no usable device, 2 usage, 3 failure (a kernel that does not finish within five seconds is one:
 * the kernel driver then resets the GPU).
 *
 * The command stream is libdrm's own compute dispatch test for this GPU generation (libdrm 2.4.115,
 * tests/amdgpu/shader_test_util.c, amdgpu_test_dispatch_memset, the gfx10 path), register for register. What
 * differs is what belongs to the kernel: its entry point, COMPUTE_PGM_RSRC1 and COMPUTE_PGM_RSRC2 as its
 * descriptor has them, the kernarg pointer as its two user SGPRs, the group size, and CS_W32_EN in the dispatch
 * for 32-lane waves.
 * That sequence is the one for compute engine version 10 (GFX10: RDNA1 and RDNA2), so anything else is refused.
 * A raw stream bypasses the Vulkan driver: run it on a GPU that drives no display (runner.py checks).
 *
 * Every submission gets command buffer memory of its own, never rewritten. Measured on a Ryzen 9 7950X3D's
 * integrated GPU: when one buffer was rewritten in place right after its fence, the second dispatch of a process
 * ran the first one's commands in 200 of 200 runs; with a fresh place for each, 24,500 distinct kernels (both
 * wave sizes) each returned their own words.
 *
 * Build: gcc -O1 hw_oracle_drm.c -o hw_oracle_drm $(pkg-config --cflags --libs libdrm_amdgpu)
 */
#include <amdgpu.h>
#include <amdgpu_drm.h>

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PACKET3(op, n) ((3u << 30) | (((op) & 0xFF) << 8) | ((n) & 0x3FFF) << 16)
#define PACKET3_COMPUTE(op, n) (PACKET3(op, n) | (1 << 1))
#define PACKET3_DISPATCH_DIRECT 0x15
#define PACKET3_SET_SH_REG 0x76
#define PACKET3_SET_UCONFIG_REG 0x79
#define PACKET3_SET_SH_REG_INDEX 0x9B

#define SENTINEL 0xDEAD0000u
#define IB_BYTES 1024u          /* one submission's commands; about 80 dwords are used */
#define IB_BLOCK (1u << 20)     /* command memory is allocated a block at a time */

struct buffer {
    amdgpu_bo_handle bo;
    amdgpu_va_handle va;
    uint64_t address;
    void *cpu;
    unsigned size;
};

static int allocate(amdgpu_device_handle dev, struct buffer *b, unsigned size, unsigned heap) {
    struct amdgpu_bo_alloc_request request = {0};
    size = (size + 4095u) & ~4095u;
    request.alloc_size = size;
    request.phys_alignment = 4096;
    request.preferred_heap = heap;
    b->size = size;
    if (amdgpu_bo_alloc(dev, &request, &b->bo)) return -1;
    if (amdgpu_va_range_alloc(dev, amdgpu_gpu_va_range_general, size, 4096, 0, &b->address, &b->va, 0)) return -1;
    if (amdgpu_bo_va_op_raw(dev, b->bo, 0, size, b->address,
                            AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE | AMDGPU_VM_PAGE_EXECUTABLE,
                            AMDGPU_VA_OP_MAP))
        return -1;
    return amdgpu_bo_cpu_map(b->bo, &b->cpu);
}

/* The LLVM processor name of a GFX10 chip, from the kernel's family id and revision (the ranges are the ones
 * Mesa's ac_gpu_info.c uses). Every GFX10 chip takes the same encodings, so this only labels the result. */
static const char *arch_name(const struct amdgpu_gpu_info *gpu) {
    const unsigned rev = gpu->chip_external_rev;
    switch (gpu->family_id) {
    case 143: /* AMDGPU_FAMILY_NV */
        if (rev < 0x0A) return "gfx1010";
        if (rev < 0x14) return "gfx1011";
        if (rev < 0x28) return "gfx1012";
        if (rev < 0x32) return "gfx1030";
        if (rev < 0x3C) return "gfx1031";
        if (rev < 0x46) return "gfx1032";
        return "gfx1034";
    case 144: return "gfx1033"; /* AMDGPU_FAMILY_VGH */
    case 146: return "gfx1035"; /* AMDGPU_FAMILY_YC */
    case 149: return "gfx1036"; /* AMDGPU_FAMILY_GC_10_3_6 */
    case 151: return "gfx1037"; /* AMDGPU_FAMILY_GC_10_3_7 */
    default: return "gfx10xx";
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: hw_oracle_drm code.bin words /dev/dri/renderD<n> < kernel-tokens\n");
        return 2;
    }
    const unsigned words = (unsigned)strtoul(argv[2], NULL, 0);
    if (words < 1 || words > 16384) { fprintf(stderr, "words must be 1..16384\n"); return 2; }

    const int fd = open(argv[3], O_RDWR);
    if (fd < 0) { perror(argv[3]); return 77; }
    uint32_t major = 0, minor = 0;
    amdgpu_device_handle dev;
    if (amdgpu_device_initialize(fd, &major, &minor, &dev)) {
        fprintf(stderr, "%s is not an AMD GPU the amdgpu kernel driver runs\n", argv[3]);
        return 77;
    }
    struct amdgpu_gpu_info gpu = {0};
    struct drm_amdgpu_info_hw_ip ip = {0};
    if (amdgpu_query_gpu_info(dev, &gpu) || amdgpu_query_hw_ip_info(dev, AMDGPU_HW_IP_COMPUTE, 0, &ip)) {
        fprintf(stderr, "device queries failed\n");
        return 77;
    }
    if (ip.hw_ip_version_major != 10 || !(ip.available_rings & 1)) {
        fprintf(stderr, "compute engine %u.%u: only version 10 (GFX10, RDNA1 and RDNA2) is supported\n",
                ip.hw_ip_version_major, ip.hw_ip_version_minor);
        return 77;
    }
    const char *name = amdgpu_get_marketing_name(dev);
    printf("# device=%s (0x%04x) arch=%s engine=%u.%u\n", name ? name : "AMD GPU", gpu.asic_id, arch_name(&gpu),
           ip.hw_ip_version_major, ip.hw_ip_version_minor);
    fflush(stdout);

    FILE *file = fopen(argv[1], "rb");
    if (!file) { perror(argv[1]); return 2; }
    fseek(file, 0, SEEK_END);
    const long code_bytes = ftell(file);
    fseek(file, 0, SEEK_SET);
    if (code_bytes < 4 || code_bytes % 4 || code_bytes > (64 << 20)) {
        fprintf(stderr, "%s: %ld bytes; expected whole dwords, at most 64 MiB\n", argv[1], code_bytes);
        return 2;
    }

    amdgpu_context_handle context;
    struct buffer cmd = {0}, code = {0}, kernarg = {0}, out = {0};
    /* The code gets a page of zeros behind it: the GPU fetches instructions ahead, past a kernel's s_endpgm. */
    if (amdgpu_cs_ctx_create(dev, &context) ||
        allocate(dev, &code, (unsigned)code_bytes + 4096, AMDGPU_GEM_DOMAIN_VRAM) ||
        allocate(dev, &kernarg, 4096, AMDGPU_GEM_DOMAIN_VRAM) || allocate(dev, &out, words * 4, AMDGPU_GEM_DOMAIN_VRAM)) {
        fprintf(stderr, "context or buffer creation failed\n");
        return 3;
    }
    memset(code.cpu, 0, code.size);
    if (fread(code.cpu, 1, (size_t)code_bytes, file) != (size_t)code_bytes) { fprintf(stderr, "short read\n"); return 2; }
    fclose(file);
    memset(kernarg.cpu, 0, kernarg.size);
    memcpy(kernarg.cpu, &out.address, 8); /* the kernel's one argument: where its output goes */

    amdgpu_bo_list_handle list = NULL;
    unsigned ib_used = IB_BLOCK; /* no block yet */

    char token[512];
    while (scanf("%511s", token) == 1) {
        /* name@offset/rsrc1/rsrc2/wave[:threads[:words]] */
        char *at = strchr(token, '@');
        if (!at) { fprintf(stderr, "bad kernel token %s (no @offset)\n", token); return 2; }
        *at++ = 0;
        char *rest = NULL;
        const unsigned long offset = strtoul(at, &rest, 0);
        uint32_t rsrc1 = 0, rsrc2 = 0;
        unsigned wave = 0, threads = 1, w = words;
        if (*rest == '/') rsrc1 = (uint32_t)strtoul(rest + 1, &rest, 0);
        if (*rest == '/') rsrc2 = (uint32_t)strtoul(rest + 1, &rest, 0);
        if (*rest == '/') wave = (unsigned)strtoul(rest + 1, &rest, 10);
        if (*rest == ':') {
            threads = (unsigned)strtoul(rest + 1, &rest, 10);
            if (*rest == ':') w = (unsigned)strtoul(rest + 1, &rest, 10);
        }
        if (*rest || offset % 256 || offset >= (unsigned long)code_bytes || (wave != 32 && wave != 64) || threads < 1 ||
            threads > 1024 || w < 1 || w > words) {
            fprintf(stderr, "bad kernel token for %s\n", token);
            return 2;
        }
        uint32_t *result = out.cpu;
        for (unsigned i = 0; i < w; i++) result[i] = SENTINEL;

        if (ib_used + IB_BYTES > IB_BLOCK) {
            /* A new block of command memory. The old one stays mapped: nothing may reuse it. */
            if (list) amdgpu_bo_list_destroy(list);
            memset(&cmd, 0, sizeof cmd);
            if (allocate(dev, &cmd, IB_BLOCK, AMDGPU_GEM_DOMAIN_GTT)) { fprintf(stderr, "command buffer allocation failed\n"); return 3; }
            amdgpu_bo_handle resources[4] = {out.bo, kernarg.bo, code.bo, cmd.bo};
            if (amdgpu_bo_list_create(dev, 4, resources, NULL, &list)) { fprintf(stderr, "bo list creation failed\n"); return 3; }
            ib_used = 0;
        }
        const uint64_t entry = code.address + offset;
        const uint64_t ib_address = cmd.address + ib_used;
        uint32_t *p = (uint32_t *)((char *)cmd.cpu + ib_used);
        ib_used += IB_BYTES;
        int i = 0;
        /* amdgpu_dispatch_init_gfx9 (no context control on the compute engine): COMPUTE_START_X..Z, TMPRING_SIZE. */
        p[i++] = PACKET3_COMPUTE(PACKET3_SET_SH_REG, 3);
        p[i++] = 0x204;
        p[i++] = 0;
        p[i++] = 0;
        p[i++] = 0;
        p[i++] = PACKET3_COMPUTE(PACKET3_SET_SH_REG, 1);
        p[i++] = 0x218;
        p[i++] = 0;
        /* amdgpu_dispatch_init_gfx10: COMPUTE_SHADER_CHKSUM, COMPUTE_REQ_CTRL (6 registers), CP_COHER_START_DELAY. */
        p[i++] = PACKET3_COMPUTE(PACKET3_SET_SH_REG, 1);
        p[i++] = 0x22a;
        p[i++] = 0;
        p[i++] = PACKET3_COMPUTE(PACKET3_SET_SH_REG, 6);
        p[i++] = 0x222;
        for (int k = 0; k < 6; k++) p[i++] = 0;
        p[i++] = PACKET3(PACKET3_SET_UCONFIG_REG, 1);
        p[i++] = 0x7b;
        p[i++] = 0x20;
        /* amdgpu_dispatch_write_cumask (gfx10): COMPUTE_STATIC_THREAD_MGMT_SE0..3. */
        p[i++] = PACKET3_COMPUTE(PACKET3_SET_SH_REG_INDEX, 2);
        p[i++] = 0x30000216;
        p[i++] = 0xffffffff;
        p[i++] = 0xffffffff;
        p[i++] = PACKET3_COMPUTE(PACKET3_SET_SH_REG_INDEX, 2);
        p[i++] = 0x30000219;
        p[i++] = 0xffffffff;
        p[i++] = 0xffffffff;
        /* amdgpu_dispatch_write2hw_gfx10: COMPUTE_PGM_LO/HI (the entry point, in 256-byte units), the shader's
         * registers, COMPUTE_PGM_RSRC3. */
        p[i++] = PACKET3_COMPUTE(PACKET3_SET_SH_REG, 2);
        p[i++] = 0x20c;
        p[i++] = (uint32_t)(entry >> 8);
        p[i++] = (uint32_t)(entry >> 40);
        const uint32_t registers[][2] = {
            {0x2e12, rsrc1},   /* COMPUTE_PGM_RSRC1, from the kernel's descriptor (0x000C0041 in the reference) */
            {0x2e13, rsrc2},   /* COMPUTE_PGM_RSRC2, from the kernel's descriptor (0x00000090 in the reference) */
            {0x2e07, threads}, /* COMPUTE_NUM_THREAD_X */
            {0x2e08, 1},       /* COMPUTE_NUM_THREAD_Y */
            {0x2e09, 1},       /* COMPUTE_NUM_THREAD_Z */
        };
        for (unsigned r = 0; r < sizeof(registers) / sizeof(registers[0]); ++r) {
            p[i++] = PACKET3_COMPUTE(PACKET3_SET_SH_REG, 1);
            p[i++] = registers[r][0] - 0x2c00;
            p[i++] = registers[r][1];
        }
        p[i++] = PACKET3_COMPUTE(PACKET3_SET_SH_REG, 1);
        p[i++] = 0x228;
        p[i++] = 0;
        /* COMPUTE_USER_DATA_0..1: the kernarg pointer. */
        p[i++] = PACKET3_COMPUTE(PACKET3_SET_SH_REG, 2);
        p[i++] = 0x240;
        p[i++] = (uint32_t)kernarg.address;
        p[i++] = (uint32_t)(kernarg.address >> 32);
        /* amdgpu_dispatch_write_dispatch_cmd: COMPUTE_RESOURCE_LIMITS, then one group. The initiator is
         * COMPUTE_SHADER_EN, with CS_W32_EN (bit 15) for 32-lane waves. */
        p[i++] = PACKET3_COMPUTE(PACKET3_SET_SH_REG, 1);
        p[i++] = 0x215;
        p[i++] = 0;
        p[i++] = PACKET3_COMPUTE(PACKET3_DISPATCH_DIRECT, 3);
        p[i++] = 1;
        p[i++] = 1;
        p[i++] = 1;
        p[i++] = 1u | (wave == 32 ? 1u << 15 : 0);
        while (i & 7) p[i++] = 0xffff1000; /* type-3 nop */

        struct amdgpu_cs_ib_info ib = {0};
        ib.ib_mc_address = ib_address;
        ib.size = (uint32_t)i;
        struct amdgpu_cs_request request = {0};
        request.ip_type = AMDGPU_HW_IP_COMPUTE;
        request.ring = 0;
        request.resources = list;
        request.number_of_ibs = 1;
        request.ibs = &ib;
        const int submitted = amdgpu_cs_submit(context, 0, &request, 1);
        if (submitted) { fprintf(stderr, "%s: amdgpu_cs_submit failed: %d\n", token, submitted); return 3; }
        struct amdgpu_cs_fence fence = {0};
        fence.context = context;
        fence.ip_type = AMDGPU_HW_IP_COMPUTE;
        fence.ring = 0;
        fence.fence = request.seq_no;
        uint32_t expired = 0;
        const int waited = amdgpu_cs_query_fence_status(&fence, 5000000000ull, 0, &expired); /* five seconds */
        if (waited || !expired) {
            fprintf(stderr, "%s did not finish within 5 s (wait returned %d): the kernel driver will time the job out "
                            "and reset this GPU\n", token, waited);
            return 3;
        }
        printf("%s", token);
        for (unsigned k = 0; k < w; k++) printf(" %08x", result[k]);
        printf("\n");
    }
    fflush(stdout);
    if (list) amdgpu_bo_list_destroy(list);
    amdgpu_cs_ctx_free(context);
    amdgpu_device_deinitialize(dev);
    close(fd);
    return 0;
}
