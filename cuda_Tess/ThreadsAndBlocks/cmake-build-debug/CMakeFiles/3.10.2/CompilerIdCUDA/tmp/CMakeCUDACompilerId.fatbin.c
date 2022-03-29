#ifndef __SKIP_INTERNAL_FATBINARY_HEADERS
#include "fatBinaryCtl.h"
#endif
#define __CUDAFATBINSECTION  ".nvFatBinSegment"
#define __CUDAFATBINDATASECTION  ".nv_fatbin"
asm(
".section .nv_fatbin, \"a\"\n"
".align 8\n"
"fatbinData:\n"
".quad 0x00100001ba55ed50,0x0000000000000330,0x0000004001010002,0x0000000000000268\n"
".quad 0x0000000000000000,0x0000001e00010007,0x0000000000000000,0x0000000000000015\n"
".quad 0x0000000000000000,0x0000000000000000,0x33010102464c457f,0x0000000000000007\n"
".quad 0x0000005b00be0002,0x0000000000000000,0x00000000000001c0,0x00000000000000c0\n"
".quad 0x00380040001e051e,0x0001000400400003,0x7472747368732e00,0x747274732e006261\n"
".quad 0x746d79732e006261,0x746d79732e006261,0x78646e68735f6261,0x666e692e766e2e00\n"
".quad 0x747368732e00006f,0x74732e0062617472,0x79732e0062617472,0x79732e006261746d\n"
".quad 0x6e68735f6261746d,0x692e766e2e007864,0x00000000006f666e,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000300000001,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000040,0x0000000000000032,0x0000000000000000\n"
".quad 0x0000000000000001,0x0000000000000000,0x000000030000000b,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000072,0x0000000000000032,0x0000000000000000\n"
".quad 0x0000000000000001,0x0000000000000000,0x0000000200000013,0x0000000000000000\n"
".quad 0x0000000000000000,0x00000000000000a8,0x0000000000000018,0x0000000000000002\n"
".quad 0x0000000000000008,0x0000000000000018,0x0000000500000006,0x00000000000001c0\n"
".quad 0x0000000000000000,0x0000000000000000,0x00000000000000a8,0x00000000000000a8\n"
".quad 0x0000000000000008,0x0000000500000001,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000008\n"
".quad 0x0000000600000001,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000008,0x0000005001010001\n"
".quad 0x0000000000000038,0x0000004000000035,0x0000001e00060001,0x0000000000000000\n"
".quad 0x0000000000002015,0x0000000000000000,0x0000000000000037,0x0000000000000048\n"
".quad 0x0000000000000000,0x762e20f000010a13,0x36206e6f69737265,0x677261742e0a312e\n"
".quad 0x30335f6d73207465,0x7365726464612e0a,0x3620657a69735f73,0x0000000a0a0a0a34\n"
".text\n");
#ifdef __cplusplus
extern "C" {
#endif
extern const unsigned long long fatbinData[104];
#ifdef __cplusplus
}
#endif
#ifdef __cplusplus
extern "C" {
#endif
static const __fatBinC_Wrapper_t __fatDeviceText __attribute__ ((aligned (8))) __attribute__ ((section (__CUDAFATBINSECTION)))= 
	{ 0x466243b1, 1, fatbinData, 0 };
#ifdef __cplusplus
}
#endif
