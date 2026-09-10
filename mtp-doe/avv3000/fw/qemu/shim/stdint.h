/* shim/stdint.h — freestanding stdint for the qemu fw build.
 *
 * This brew arm-none-eabi-gcc ships without newlib (no target stdint.h
 * behind gcc's include_next wrapper), and the fixture's qemu suite is
 * 100% freestanding anyway (Unity compiled with UNITY_EXCLUDE_* + our
 * own code). Shadow gcc's wrapper with this via -isystem.
 */
#ifndef SHIM_STDINT_H
#define SHIM_STDINT_H

typedef unsigned char       uint8_t;
typedef signed char         int8_t;
typedef unsigned short      uint16_t;
typedef signed short        int16_t;
typedef unsigned int        uint32_t;
typedef signed int          int32_t;
typedef unsigned long long  uint64_t;
typedef signed long long    int64_t;
typedef unsigned int        uintptr_t;   /* 32-bit ARM */
typedef signed int          intptr_t;
typedef unsigned long long  uintmax_t;
typedef signed long long    intmax_t;

#define INT8_MIN   (-128)
#define INT8_MAX   127
#define UINT8_MAX  255
#define INT16_MIN  (-32767 - 1)
#define INT16_MAX  32767
#define UINT16_MAX 65535
#define INT32_MIN  (-2147483647 - 1)
#define INT32_MAX  2147483647
#define UINT32_MAX 0xffffffffUL
#define INT64_MIN  (-9223372036854775807LL - 1)
#define INT64_MAX  9223372036854775807LL
#define UINT64_MAX 0xffffffffffffffffULL
#define INTPTR_MIN INT32_MIN
#define INTPTR_MAX INT32_MAX
#define UINTPTR_MAX UINT32_MAX
#define INTMAX_MIN INT64_MIN
#define INTMAX_MAX INT64_MAX
#define UINTMAX_MAX UINT64_MAX
#define SIZE_MAX   UINT32_MAX
#define PTRDIFF_MIN INT32_MIN
#define PTRDIFF_MAX INT32_MAX

#endif /* SHIM_STDINT_H */
