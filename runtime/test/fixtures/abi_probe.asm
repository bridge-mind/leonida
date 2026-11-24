; Test fixture for unimplemented_policy_test: calls a function with every nonvolatile register set to a known pattern
; and records what each holds when the call returns, so the test can tell whether the callee kept the Win64 ABI.
;
;   uint64_t abi_probe_call(AbiProbe* p, void* fn)      p must be 16-byte aligned
;
;   struct AbiProbe {            offset
;       uint64_t gpr_in[8];      0x000   rbx rbp rdi rsi r12 r13 r14 r15, loaded before the call
;       uint8_t  xmm_in[10][16]; 0x040   xmm6..xmm15, loaded before the call
;       uint64_t gpr_out[8];     0x100   the same registers after the call
;       uint8_t  xmm_out[10][16];0x140
;       uint64_t result;         0x1E0   rax after the call
;   };
; rcx..r9 are 1..4 at the call. The function's own frame saves and restores what it uses, and carries unwind data, so
; a stack walk can pass through it.

.code

abi_probe_call PROC FRAME
    push rbx
    .pushreg rbx
    push rbp
    .pushreg rbp
    push rdi
    .pushreg rdi
    push rsi
    .pushreg rsi
    push r12
    .pushreg r12
    push r13
    .pushreg r13
    push r14
    .pushreg r14
    push r15
    .pushreg r15
    sub rsp, 0D8h
    .allocstack 0D8h
    movaps xmmword ptr [rsp+30h], xmm6
    .savexmm128 xmm6, 30h
    movaps xmmword ptr [rsp+40h], xmm7
    .savexmm128 xmm7, 40h
    movaps xmmword ptr [rsp+50h], xmm8
    .savexmm128 xmm8, 50h
    movaps xmmword ptr [rsp+60h], xmm9
    .savexmm128 xmm9, 60h
    movaps xmmword ptr [rsp+70h], xmm10
    .savexmm128 xmm10, 70h
    movaps xmmword ptr [rsp+80h], xmm11
    .savexmm128 xmm11, 80h
    movaps xmmword ptr [rsp+90h], xmm12
    .savexmm128 xmm12, 90h
    movaps xmmword ptr [rsp+0A0h], xmm13
    .savexmm128 xmm13, 0A0h
    movaps xmmword ptr [rsp+0B0h], xmm14
    .savexmm128 xmm14, 0B0h
    movaps xmmword ptr [rsp+0C0h], xmm15
    .savexmm128 xmm15, 0C0h
    .endprolog

    mov [rsp+20h], rcx
    mov r11, rdx

    mov rbx, [rcx+00h]
    mov rbp, [rcx+08h]
    mov rdi, [rcx+10h]
    mov rsi, [rcx+18h]
    mov r12, [rcx+20h]
    mov r13, [rcx+28h]
    mov r14, [rcx+30h]
    mov r15, [rcx+38h]
    movaps xmm6,  xmmword ptr [rcx+40h]
    movaps xmm7,  xmmword ptr [rcx+50h]
    movaps xmm8,  xmmword ptr [rcx+60h]
    movaps xmm9,  xmmword ptr [rcx+70h]
    movaps xmm10, xmmword ptr [rcx+80h]
    movaps xmm11, xmmword ptr [rcx+90h]
    movaps xmm12, xmmword ptr [rcx+0A0h]
    movaps xmm13, xmmword ptr [rcx+0B0h]
    movaps xmm14, xmmword ptr [rcx+0C0h]
    movaps xmm15, xmmword ptr [rcx+0D0h]

    mov ecx, 1
    mov edx, 2
    mov r8d, 3
    mov r9d, 4
    call r11

    mov rcx, [rsp+20h]
    mov [rcx+1E0h], rax
    mov [rcx+100h], rbx
    mov [rcx+108h], rbp
    mov [rcx+110h], rdi
    mov [rcx+118h], rsi
    mov [rcx+120h], r12
    mov [rcx+128h], r13
    mov [rcx+130h], r14
    mov [rcx+138h], r15
    movaps xmmword ptr [rcx+140h], xmm6
    movaps xmmword ptr [rcx+150h], xmm7
    movaps xmmword ptr [rcx+160h], xmm8
    movaps xmmword ptr [rcx+170h], xmm9
    movaps xmmword ptr [rcx+180h], xmm10
    movaps xmmword ptr [rcx+190h], xmm11
    movaps xmmword ptr [rcx+1A0h], xmm12
    movaps xmmword ptr [rcx+1B0h], xmm13
    movaps xmmword ptr [rcx+1C0h], xmm14
    movaps xmmword ptr [rcx+1D0h], xmm15

    movaps xmm6,  xmmword ptr [rsp+30h]
    movaps xmm7,  xmmword ptr [rsp+40h]
    movaps xmm8,  xmmword ptr [rsp+50h]
    movaps xmm9,  xmmword ptr [rsp+60h]
    movaps xmm10, xmmword ptr [rsp+70h]
    movaps xmm11, xmmword ptr [rsp+80h]
    movaps xmm12, xmmword ptr [rsp+90h]
    movaps xmm13, xmmword ptr [rsp+0A0h]
    movaps xmm14, xmmword ptr [rsp+0B0h]
    movaps xmm15, xmmword ptr [rsp+0C0h]
    add rsp, 0D8h
    pop r15
    pop r14
    pop r13
    pop r12
    pop rsi
    pop rdi
    pop rbp
    pop rbx
    ret
abi_probe_call ENDP

END
