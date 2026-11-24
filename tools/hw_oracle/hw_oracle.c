/* Hardware oracle launcher. Loads a code object with the HIP runtime that ships inside the AMD Windows driver
 * (amdhip64_7.dll; no SDK, headers or import library) and runs kernels one at a time.
 *
 *   hw_oracle <code.co> <words> [hip.dll]   kernel names on stdin, whitespace separated; the DLL defaults to
 *                                           amdhip64_7.dll (then _6) from the normal search path
 *
 * Each kernel gets one 8-byte kernarg (a pointer to a <words> dword buffer pre-filled with 0xDEAD0000) and is
 * launched as one workgroup of one thread. A stdin token may be name:threads:words to launch a bigger workgroup
 * and read back fewer words (threads 1..1024, words <= <words>). Output, one line per kernel:  <name> <hex word> ...
 * The first line is "# device=<name> arch=<gfxNNNN>". Exit 0 ok, 77 no runtime or no device, 2 usage, 3 failure.
 * Build: gcc -O1 hw_oracle.c -o hw_oracle.exe
 */
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#define LOAD(n) n##_t n = (n##_t)GetProcAddress(h, #n); if (!n) { fprintf(stderr, "missing " #n "\n"); return 77; }
typedef int (*hipInit_t)(unsigned);
typedef int (*hipGetDeviceCount_t)(int *);
typedef int (*hipDeviceGetName_t)(char *, int, int);
typedef int (*hipGetDevicePropertiesR0600_t)(void *, int);
typedef int (*hipModuleLoadData_t)(void **, const void *);
typedef int (*hipModuleGetFunction_t)(void **, void *, const char *);
typedef int (*hipMalloc_t)(void **, size_t);
typedef int (*hipMemcpy_t)(void *, const void *, size_t, int);
typedef int (*hipDeviceSynchronize_t)(void);
typedef int (*hipModuleLaunchKernel_t)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                       void *, void **, void **);
#define CK(x) do { int e_ = (x); if (e_) { fprintf(stderr, "%s -> hip error %d\n", #x, e_); return 3; } } while (0)

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: hw_oracle code.co words < kernel-names\n"); return 2; }
    unsigned words = (unsigned)strtoul(argv[2], NULL, 0);
    if (words < 1 || words > 16384) { fprintf(stderr, "words must be 1..16384\n"); return 2; }
    HMODULE h = NULL;
    if (argc > 3) {
        h = LoadLibraryA(argv[3]);
        if (!h) { fprintf(stderr, "cannot load %s\n", argv[3]); return 77; }
    } else {
        h = LoadLibraryA("amdhip64_7.dll");
        if (!h) h = LoadLibraryA("amdhip64_6.dll");
    }
    if (!h) { fprintf(stderr, "no amdhip64 DLL (AMD driver not installed)\n"); return 77; }
    LOAD(hipInit) LOAD(hipGetDeviceCount) LOAD(hipDeviceGetName) LOAD(hipGetDevicePropertiesR0600) LOAD(hipModuleLoadData)
    LOAD(hipModuleGetFunction) LOAD(hipMalloc) LOAD(hipMemcpy) LOAD(hipDeviceSynchronize) LOAD(hipModuleLaunchKernel)
    int n = 0;
    if (hipInit(0) || hipGetDeviceCount(&n) || n < 1) { fprintf(stderr, "no AMD device\n"); return 77; }
    char name[256] = {0};
    CK(hipDeviceGetName(name, 256, 0));
    static char prop[16384];
    memset(prop, 0, sizeof prop);
    hipGetDevicePropertiesR0600(prop, 0);
    const char *arch = "?";
    for (int i = 0; i < (int)sizeof prop - 8; i++) if (!memcmp(prop + i, "gfx", 3)) { arch = prop + i; break; }
    printf("# device=%s arch=%s\n", name, arch);
    fflush(stdout);
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("code object"); return 2; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    void *img = malloc(sz);
    if (fread(img, 1, sz, f) != (size_t)sz) { fprintf(stderr, "short read\n"); return 2; }
    fclose(f);
    void *mod;
    CK(hipModuleLoadData(&mod, img));   /* 209 = no binary for this GPU: wrong -mcpu, or no .amdgpu_metadata */
    void *d;
    CK(hipMalloc(&d, words * 4));
    uint32_t *buf = malloc(words * 4);
    char kn[256];
    while (scanf("%255s", kn) == 1) {
        /* token: name[:threads[:words]] */
        unsigned threads = 1, w = words;
        char *c = strchr(kn, ':');
        if (c) {
            *c++ = 0;
            threads = (unsigned)strtoul(c, &c, 10);
            if (*c == ':') w = (unsigned)strtoul(c + 1, NULL, 10);
            if (threads < 1 || threads > 1024 || w < 1 || w > words) { fprintf(stderr, "bad kernel token\n"); return 2; }
        }
        for (unsigned i = 0; i < w; i++) buf[i] = 0xDEAD0000u;
        CK(hipMemcpy(d, buf, w * 4, 1));
        void *fn;
        CK(hipModuleGetFunction(&fn, mod, kn));
        void *arg = d;
        size_t asz = sizeof arg;
        void *extra[] = {(void *)1, &arg, (void *)2, &asz, (void *)3};
        CK(hipModuleLaunchKernel(fn, 1, 1, 1, threads, 1, 1, 0, NULL, NULL, extra));
        CK(hipDeviceSynchronize());
        CK(hipMemcpy(buf, d, w * 4, 2));
        printf("%s", kn);
        for (unsigned i = 0; i < w; i++) printf(" %08x", buf[i]);
        printf("\n");
    }
    fflush(stdout);
    return 0;
}
