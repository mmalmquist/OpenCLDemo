#include "util.h"

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <time.h>

#define BUFFER_SIZE 0x1000

extern char *
read_file(FILE *fp,
	  size_t *src_size)
{
  if (!fp) {
    return NULL;
  }
  
  size_t src_offset = 0, max_len = BUFFER_SIZE;
  char *src_str = (char *) calloc(max_len, sizeof(char));
  assert(src_str);
  
  char *buffer = (char *) calloc(BUFFER_SIZE, sizeof(char));
  assert(buffer);
  
  for (size_t bytes_read; (bytes_read = fread(buffer, sizeof(char), BUFFER_SIZE, fp)) > 0;) {
    if (src_offset + bytes_read >= max_len) {
      max_len = (3*max_len)/2;
      src_str = (char *) realloc(src_str, max_len*sizeof(char));
      assert(src_str);
    }
    memcpy(src_str + src_offset, buffer, bytes_read);
    src_offset += bytes_read;
  }
  if (src_size) {
    *src_size = src_offset;
  }
  return src_str;
}

extern double
get_time(void)
{
  struct timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  return t.tv_sec + t.tv_nsec/1e9;
}
