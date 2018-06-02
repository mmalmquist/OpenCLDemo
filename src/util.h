#ifndef UTIL_H_
#define UTIL_H_

#include <stdio.h>

extern char *
read_file(FILE *fp,
	  size_t *src_size);

extern double
get_time(void);

#endif //UTIL_H_
