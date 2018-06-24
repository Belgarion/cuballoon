
#include "sha256.h"

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define NBLOCKS 4

__attribute__((aligned(16)))
static __sha256_block_t blocks[NBLOCKS];

__attribute__((aligned(16)))
static __sha256_hash_t hash[2][NBLOCKS];


static const uint32_t __sha256_init[] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};


/* Reference implementation taken from opensolaris */
extern void __sha256_osol(const __sha256_block_t blk, __sha256_hash_t hash);
void sha256_osol(int result)
{
    for (int i = 0; i < NBLOCKS; ++i) {
        __builtin_memcpy(hash[result][i], __sha256_init, 32);
        __sha256_osol(blocks[i], hash[result][i]);
    }
}

extern void __sha256_int(__sha256_block_t *blk[4], __sha256_hash_t *hash[4]);
void sha256_int(int result)
{
    for (int i = 0; i < NBLOCKS; ++i) {
        __builtin_memcpy(hash[result][i], __sha256_init, 32);
    }

    __sha256_block_t *blk[4] = { &blocks[0], &blocks[1], &blocks[2], &blocks[3] };
    __sha256_hash_t *hsh[4] = { &hash[result][0], &hash[result][1], &hash[result][2], &hash[result][3] };
    __sha256_int(blk, hsh);
    
    //if (memcmp(&hash[result][0], &hash[result][1], 8*4)) {
    //    printf("error\n");
    //}
}



#define SHA256_HASH_SIZE	(8)
#define SHA256_BLOCK_SIZE	(16)

void dump_H(const uint32_t *h)
{
    int i;
    
    for (i = 0; i < SHA256_HASH_SIZE; ++i) {
        printf(" 0x%08x", h[i]);
    }
}

void one(unsigned n)
{
	unsigned i, j;
	struct timeval tv_start, tv_end;
	double delta;
	double best;
	unsigned n_iter;
    
	n_iter =  1000*(8192/n);
	best = UINT32_MAX;
	for (j = 0; j < 10; ++j) {
		gettimeofday(&tv_start, 0);
		for (i = 0; i < n_iter; ++i) {
			sha256_osol(0);
		}
		gettimeofday(&tv_end, 0);
        
		__asm volatile("emms");
        
		delta = (double)(tv_end.tv_sec - tv_start.tv_sec)
        + (double)(tv_end.tv_usec - tv_start.tv_usec) / 1000000.0;
		if (delta < best) {
			best = delta;
		}
	}
	/* print a number similar to what openssl reports */
    printf("%.2f blocks per second\n", (double)(4 * n_iter) / best / 1000.0 + 0.005);
    
    n_iter =  1000*(8192/n);
	best = UINT32_MAX;
	for (j = 0; j < 10; ++j) {
		gettimeofday(&tv_start, 0);
		for (i = 0; i < n_iter; ++i) {
			sha256_int(0);
		}
		gettimeofday(&tv_end, 0);
        
		__asm volatile("emms");
        
		delta = (double)(tv_end.tv_sec - tv_start.tv_sec)
        + (double)(tv_end.tv_usec - tv_start.tv_usec) / 1000000.0;
		if (delta < best) {
			best = delta;
		}
	}
	/* print a number similar to what openssl reports */
    printf("%.2f blocks per second\n", (double)(4 * n_iter) / best / 1000.0 + 0.005);
}

static void test()
{
    srand(time(NULL));
    uint8_t *ptr = (uint8_t *) blocks;
    for (int i = 0; i < NBLOCKS * sizeof(__sha256_block_t); ++i) {
        *ptr++ = rand();
    }
    
    sha256_osol(0);
    sha256_int(1);
    
    for (int i = 0; i < NBLOCKS; ++i) {
        if (memcmp(hash[0][i], hash[1][i], sizeof(__sha256_hash_t))) {
            printf("FAILED at %d\n", i);
            dump_H(hash[0][i]); printf("\n");
            dump_H(hash[1][i]); printf("\n");
        }
    }
}

int main(int argc, const char * argv[])
{
	test();
	one(1024);



	__attribute__((aligned(16)))
	static __sha256_block_t block[3];
	unsigned char ds[168] = "\x00\x10\x00\x00\x00\x00\x00\x00\x29\xc4\xfb\x2f\x2d\x8a\x05\xd3\x3d\x8d\x1d\xb1\x34\x18\x90\x23\x31\xba\xe9\x68\xcb\x1b\xe6\xc1\xfe\x82\x93\xe5\xd2\x23\x08\xc7\x0a\x33\x05\xf2\xc0\x7a\xfd\x72\xe6\x0a\x6e\xad\x25\x53\x3e\x5d\x4f\x9e\x95\xe3\x06\x89\x5d\x6e\x26\xd8\xac\xe9\xd4\x86\xb9\x86\x18\x31\x36\x31\xb8\xc4\x3b\x12\x80\x6f\x64\xcd\xa2\x31\x7e\x48\x4a\xe5\xdb\xd5\xde\x03\xb2\xb6\x81\xe1\xb5\xf5\x31\x27\xa9\x32\xda\x84\x1f\x78\x30\x81\x20\xfd\x5f\x1d\x63\x74\x3f\x69\x31\x32\x2e\x7b\xad\x2d\xb6\xc0\xdf\x9c\xc6\xd6\x86\x44\x0b\x49\xa3\xcd\x74\x83\x16\x86\xea\x47\x85\x88\x45\x84\x6c\xec\x67\x0a\xf8\x34\x31\xf6\x2c\x76\xfa\x2f\x51\x2a\xee\x7b\xcd\x3a\xdf\x1c\xbf\xcb";
	uint8_t *ptr = (uint8_t*)block;
	printf("data: ");
	// 60 bytes of data
	for (int i = 0; i < 168; i++) {
		*ptr++ = ds[i];
	}

	*ptr++ = 0x80; // Padding end of string marker
	// Pad to (k+l+1 = 448 mod 512)
	// l = 168*8 = 1344bits
	// Blocks: 512bit | 512bit | 512bit
	// (512*3-65-l) = 1536-65-l = 1471 - l = 1471-1344 = 127bit = 15.875 bytes
	for (int i = 0; i < 15; i++) {
		*ptr++ = 0;
	}
	// 8 bytes is length (in bits)
	// 480bit = 0x1e0
	*ptr++ = 0x0;
	*ptr++ = 0;
	*ptr++ = 0;
	*ptr++ = 0;
	*ptr++ = 0;
	*ptr++ = 0;
	*ptr++ = 0x5;
	*ptr++ = 0x40;

	/*__sha256_block_t *nblk[4] = { &block, &block, &block, &block };

	__attribute__((aligned(16)))
	static __sha256_hash_t nhash[NBLOCKS];
	__builtin_memcpy(nhash, __sha256_init, 32);

	__sha256_hash_t *hsh[4] = { &nhash[0], &nhash[1], &nhash[2], &nhash[3] };
	printf("sha256_int\n");
	__sha256_int(nblk, hsh);
	printf("dump_h\n");
	dump_H(*hsh[0]); printf("\n");*/

	printf("data: ");
	for (int i = 0; i < 168; i++) {
		printf("\\x%02x", ((uint8_t*)block)[i]);
	}
	printf("\n");

	static __sha256_hash_t ohash;
    __builtin_memcpy(ohash, __sha256_init, 32);
    __sha256_osol(block[0], ohash);
    __sha256_osol(block[1], ohash);
    __sha256_osol(block[2], ohash);
	dump_H(ohash); printf("\n");


	printf("DONE");

    return 0;
}

