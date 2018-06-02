#ifndef __COMMON_H_
#define __COMMON_H_

#include <stdio.h>


//#define __DEBUG_ENABLED__
#define __PRINT_RESULT__

#if defined __PRINT_RESULT__
 #define PRINRM(fmt, ...) printf(fmt, __VA_ARGS__)
#else
 #define PRINRM(fmt, ...)
#endif
#define PRIINFO(fmt, ...) printf("[INFO] " fmt, __VA_ARGS__)

#if defined __DEBUG_ENABLED__
 #define PRIDBG(fmt, ...) printf("[DEBUG] " fmt, __VA_ARGS__)
 #define PRIDBG_WHEN(cond, fmt, ...) if (cond) printf("[DEBUG] " fmt, __VA_ARGS__);
#else
 #define PRIDBG(fmt, ...)
 #define PRIDBG_WHEN(cond, fmt, ...)
#endif

#define MIN(A, B) ((A) < (B) ? (A) : (B))
#define MAX(A, B) ((A) > (B) ? (A) : (B))

#endif
