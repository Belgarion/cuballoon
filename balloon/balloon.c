#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <openssl/aes.h>
#include <openssl/evp.h>
#include <openssl/sha.h>
#include "balloon.h"
#include <sys/time.h>
#include "miner.h"
#include "../sha256-sse/sha256.h"

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

#ifdef __cplusplus
extern "C"{
#endif

static const uint32_t __sha256_init[] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};



void balloon_init (struct balloon_options *opts, int64_t s_cost, int32_t t_cost) {
  opts->s_cost = s_cost;
  opts->t_cost = t_cost;
}

void balloon_128 (unsigned char *input, unsigned char *output) {
  balloon (input, output, 80, 128, 4);
}

void balloon_hash (unsigned char *input, unsigned char *output, int64_t s_cost, int32_t t_cost) {
  balloon (input, output, 80, s_cost, t_cost);
}

void balloon (unsigned char *input, unsigned char *output, int32_t len, int64_t s_cost, int32_t t_cost) {
  struct balloon_options opts;
  struct hash_state s;
  struct timeval tv1,tv2,tv3,tv4,tv5,tv6;
  gettimeofday(&tv1, NULL);
  balloon_init (&opts, s_cost, t_cost);
  gettimeofday(&tv2, NULL);
  hash_state_init (&s, &opts, input);
  gettimeofday(&tv3, NULL);
  hash_state_fill (&s, input, len);
  gettimeofday(&tv4, NULL);
  hash_state_mix (&s, t_cost);
  gettimeofday(&tv5, NULL);
  hash_state_extract (&s, output);
  gettimeofday(&tv6, NULL);
  hash_state_free (&s);
  double init = (double)(tv2.tv_usec - tv1.tv_usec) / 1000000 + (double)(tv2.tv_sec - tv1.tv_sec);
  double state_init = (double)(tv3.tv_usec - tv2.tv_usec) / 1000000 + (double)(tv3.tv_sec - tv2.tv_sec);
  double state_fill = (double)(tv4.tv_usec - tv3.tv_usec) / 1000000 + (double)(tv4.tv_sec - tv3.tv_sec);
  double state_mix = (double)(tv5.tv_usec - tv4.tv_usec) / 1000000 + (double)(tv5.tv_sec - tv4.tv_sec);
  double state_extract = (double)(tv6.tv_usec - tv5.tv_usec) / 1000000 + (double)(tv6.tv_sec - tv5.tv_sec);
  static int printcnt = 0;
  printcnt++;
  if (printcnt >= 15) {
	  printf("Balloon timing: init: %.8f, state_init: %.8f, state_fill: %.8f, state_mix: %.8f, state_extract: %.8f\n",
			  init, state_init, state_fill, state_mix, state_extract);
	  printcnt = 0;
  }
}

static inline int bitstream_init (struct bitstream *b) {
  SHA256_Init(&b->c);
  b->initialized = false;
#if   OPENSSL_VERSION_NUMBER >= 0x10100000L
  b->ctx = EVP_CIPHER_CTX_new();
  EVP_CIPHER_CTX_init(b->ctx);
#else
  EVP_CIPHER_CTX_init (&b->ctx);
#endif
  b->zeros = malloc (BITSTREAM_BUF_SIZE * sizeof (uint8_t));
  memset (b->zeros, 0, BITSTREAM_BUF_SIZE);
}

static inline int bitstream_free (struct bitstream *b) {
  uint8_t out[AES_BLOCK_SIZE];
  int outl;
#if   OPENSSL_VERSION_NUMBER >= 0x10100000L
  EVP_EncryptFinal (b->ctx, out, &outl);
  EVP_CIPHER_CTX_cleanup (b->ctx);
  EVP_CIPHER_CTX_free(b->ctx);
#else
  EVP_EncryptFinal (&b->ctx, out, &outl);
  EVP_CIPHER_CTX_cleanup (&b->ctx);
#endif
  free (b->zeros);
}

static inline int bitstream_seed_add (struct bitstream *b, const void *seed, size_t seedlen) {
  SHA256_Update(&b->c, seed, seedlen);
}

int bitstream_seed_finalize (struct bitstream *b) {
  uint8_t key_bytes[SHA256_DIGEST_LENGTH];
  SHA256_Final (key_bytes, &b->c);
  uint8_t iv[AES_BLOCK_SIZE];
  memset (iv, 0, AES_BLOCK_SIZE);
  /*
  printf("EVP_EncryptInit(b->ctx, EVP_aes_128_ctr(), key_bytes=");
  for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
	  printf("%x ", key_bytes[i]);
  }
  printf(", iv=");
  for (int i = 0; i < AES_BLOCK_SIZE; i++) {
	  printf("%x ", iv[i]);
  }
  printf("\n");*/
#if   OPENSSL_VERSION_NUMBER >= 0x10100000L
  EVP_CIPHER_CTX_set_padding (b->ctx, 1);
  EVP_EncryptInit (b->ctx, EVP_aes_128_ctr (), key_bytes, iv);
#else
  EVP_CIPHER_CTX_set_padding (&b->ctx, 1);
  EVP_EncryptInit (&b->ctx, EVP_aes_128_ctr (), key_bytes, iv);
#endif
  b->initialized = true;
}

static inline void encrypt_partial (struct bitstream *b, void *outp, int to_encrypt) {
  int encl;
#if   OPENSSL_VERSION_NUMBER >= 0x10100000L
  EVP_EncryptUpdate (b->ctx, outp, &encl, b->zeros, to_encrypt);
#else
  EVP_EncryptUpdate (&b->ctx, outp, &encl, b->zeros, to_encrypt);
#endif
}

inline int bitstream_fill_buffer (struct bitstream *b, void *out, size_t outlen) {
  size_t total = 0;
  while (total < outlen) {
    const int to_encrypt = MIN(outlen - total, BITSTREAM_BUF_SIZE);
    encrypt_partial (b, (void*)((uint8_t*)out + total), to_encrypt);
    total += to_encrypt;
  }
}

//#define DEBUG_COMPRESS
static inline void compress (uint64_t *counter, uint8_t *out, const uint8_t *blocks[], size_t blocks_to_comp) {
  SHA256_CTX ctx;
  SHA256_Init (&ctx);
  SHA256_Update (&ctx, counter, 8);
#ifdef DEBUG_COMPRESS
  printf("[compress] shadata: ");
  const unsigned char *ctr = (void*)counter;
  for (int i = 0; i < 8; i++) {
  	printf("\\x%02x", *(ctr++));
  }
#endif
  for (unsigned int i = 0; i < blocks_to_comp; i++) {
    SHA256_Update (&ctx, *(blocks+i), BLOCK_SIZE);
#ifdef DEBUG_COMPRESS
	unsigned char *b = *(blocks+i);
	for (int j = 0; j < BLOCK_SIZE; j++) {
		printf("\\x%02x", b[j]);
	}
#endif
  }
#ifdef DEBUG_COMPRESS
  printf("\n");
#endif
  SHA256_Final (out, &ctx);
#ifdef DEBUG_COMPRESS
  printf("[compress] sha256hash: ");
  for (int i = 0; i < 32; i++) {
	  printf("%02x ", *(out+i));
  }
  printf("\n");
#endif
  *counter += 1;
}

static inline int expand (uint64_t *counter, uint8_t *buf, size_t blocks_in_buf) {
  const uint8_t *blocks[1] = { buf };
  uint8_t *cur = buf + BLOCK_SIZE;
  for (size_t i = 1; i < blocks_in_buf; i++) { 
    compress (counter, cur, blocks, 1);
    *blocks += BLOCK_SIZE;
    cur += BLOCK_SIZE;
  }
}

static inline uint64_t bytes_to_littleend_uint64 (const uint8_t *bytes, size_t n_bytes) {
  if (n_bytes > 8) 
    n_bytes = 8;
  uint64_t out = 0;
  for (int i = n_bytes-1; i >= 0; i--) {
    out <<= 8;
    out |= bytes[i];
  }
  return out;
}
inline void bytes_to_littleend8_uint64 (const uint8_t *bytes, uint64_t *out) {
	*out <<= 8;
	*out |= *(bytes + 7);
	*out <<= 8;
	*out |= *(bytes + 6);
	*out <<= 8;
	*out |= *(bytes + 5);
	*out <<= 8;
	*out |= *(bytes + 4);
	*out <<= 8;
	*out |= *(bytes + 3);
	*out <<= 8;
	*out |= *(bytes + 2);
	*out <<= 8;
	*out |= *(bytes + 1);
	*out <<= 8;
	*out |= *(bytes + 0);
}

static inline void * block_index (const struct hash_state *s, size_t i) {
  return s->buffer + (BLOCK_SIZE * i);
}

static inline uint64_t options_n_blocks (const struct balloon_options *opts) {
  const uint32_t bsize = BLOCK_SIZE;
  uint64_t ret = (opts->s_cost * 1024) / bsize;
  return (ret < BLOCKS_MIN) ? BLOCKS_MIN : ret;
}

static inline void * block_last (const struct hash_state *s) {
  return block_index (s, s->n_blocks - 1);
}

inline int hash_state_init (struct hash_state *s, const struct balloon_options *opts, const uint8_t salt[SALT_LEN]) {
  s->counter = 0;
  s->n_blocks = options_n_blocks (opts);
  if (s->n_blocks % 2 != 0) s->n_blocks++;
  s->has_mixed = false;
  s->opts = opts;
  s->buffer = malloc (s->n_blocks * BLOCK_SIZE);
  int a = salt[0];
  a++;
  bitstream_init (&s->bstream);
  bitstream_seed_add (&s->bstream, salt, SALT_LEN);
#ifdef DEBUG
  printf("salt: ");
  for (int i = 0; i < SALT_LEN; i++) {
	  printf("%x ", salt[i]);
  }
  printf("\n");
#endif
  bitstream_seed_add (&s->bstream, &opts->s_cost, 8);
  bitstream_seed_add (&s->bstream, &opts->t_cost, 4);
  bitstream_seed_finalize (&s->bstream);
}

int hash_state_free (struct hash_state *s) {
  bitstream_free (&s->bstream);
  free (s->buffer);
}

inline int hash_state_fill (struct hash_state *s, const uint8_t *in, size_t inlen) {
  SHA256_CTX c;
  SHA256_Init (&c);
  SHA256_Update (&c, &s->counter, 8);
  SHA256_Update (&c, in, SALT_LEN);
  SHA256_Update (&c, in, inlen);
  SHA256_Update (&c, &s->opts->s_cost, 8);
  SHA256_Update (&c, &s->opts->t_cost, 4);
  SHA256_Final (s->buffer, &c);
  s->counter++;
  expand (&s->counter, s->buffer, s->n_blocks);
}


//#define DEBUG_SHA256
extern void __sha256_osol(const __sha256_block_t blk, __sha256_hash_t hash);
static void sha256_168byte(uint8_t *data, uint8_t *outhash) {
	// outhash should be 32 byte
	//
	// l = 168byte => 1344bit (requires 3 blocks)
	// (k + 1 + l) mod 512 = 448
	// 512 * 3 = 1536 >= 1344:
	// k = 3*512 - 65 - l = 1536 - 65 - 1344 = 127 bits of padding => 15.875 bytes 	

	//__attribute__((aligned(16)))
	static __sha256_block_t block[3];
	static int block_inited = 0;
	uint8_t *ptr = (uint8_t*)block;
	// 168 bytes of data
	memcpy(ptr, data, 168);
	ptr += 168;
	/*for (int i = 0; i < 168; i++) {
		*ptr++ = data[i];
	}*/

#ifdef DEBUG_SHA256
	printf("256data: ");
	for (int i = 0; i < 168; i++) {
		printf("\\x%02x", ((uint8_t*)block)[i]);
	}
	printf("\n");
#endif

	if (!block_inited) {
		*ptr++ = 0x80; // End of string marker (and 7 bits padding)
		// Pad to (k+l+1 = 448 mod 512)
		// l = 168*8 = 1344bits
		// Blocks: 512bit | 512bit | 512bit
		// (512*3-65-l) = 1536-65-l = 1471 - l = 1471-1344 = 127bit = 15.875 bytes
		/*for (int i = 0; i < 15; i++) {
			*ptr++ = 0;
		}*/
		memset(ptr, 0, 15);
		ptr += 15;
		// 8 bytes is length (in bits)
		// 1344bit = 0x540
		*ptr++ = 0x0;
		*ptr++ = 0;
		*ptr++ = 0;
		*ptr++ = 0;
		*ptr++ = 0;
		*ptr++ = 0;
		*ptr++ = 0x5;
		*ptr++ = 0x40;

		block_inited = 1;
	}

	__sha256_hash_t ohash;
	memcpy(ohash, __sha256_init, 32);
	__sha256_osol(block[0], ohash);
	__sha256_osol(block[1], ohash);
	__sha256_osol(block[2], ohash);

	uint8_t *h = ohash;
	uint8_t *outp = outhash;
#ifdef DEBUG_SHA256
	printf("sha256_168byte hash: ");
#endif
	for (int i = 0; i < 32/4; i++) {
#ifdef DEBUG_SHA256
		printf("%02x ", h[3]);
		printf("%02x ", h[2]);
		printf("%02x ", h[1]);
		printf("%02x ", h[0]);
#endif
		// Fix endianness at the same time
		*outp++ = h[3];
		*outp++ = h[2];
		*outp++ = h[1];
		*outp++ = h[0];
		h += 4;
	}
#ifdef DEBUG_SHA256
	printf("\n");
#endif

}
#include <unistd.h>

uint8_t prebuf[409600];
uint64_t prebuf_le[409600 / 8];
uint8_t prebuf_filled = 0;
void hash_state_mix (struct hash_state *s, int32_t mixrounds) {
	if (!prebuf_filled) {
		bitstream_fill_buffer (&s->bstream, prebuf, 409600);
		prebuf_filled = 1;
		uint8_t *buf = prebuf;
		uint64_t *lebuf = prebuf_le;
		for (int i = 0; i < 409600; i+=8) {
			bytes_to_littleend8_uint64(buf, lebuf);
			*lebuf %= 4096;
			lebuf++;
			buf += 8;
		}
	}
	uint64_t *buf = prebuf_le;
	uint8_t *sbuf = s->buffer;

	uint64_t neighbor;
	int32_t n_blocks = s->n_blocks;
	uint8_t *last_block = (sbuf + (BLOCK_SIZE*(n_blocks-1)));
	for (int32_t rounds=0; rounds < mixrounds; rounds++) {
		static const uint8_t *blocks[5];
		uint8_t **block = (uint8_t**)blocks;
		static unsigned char data[8 + BLOCK_SIZE*5];
		static unsigned char *db1 = data+8;
		static unsigned char *db2 = data+40;
		static unsigned char *db3 = data+72;
		static unsigned char *db4 = data+104;
		static unsigned char *db5 = data+136;
		{ // i = 0
			blocks[0] = last_block;
			blocks[1] = sbuf;
			blocks[2] = (sbuf + (BLOCK_SIZE * (*(buf++))));
			blocks[3] = (sbuf + (BLOCK_SIZE * (*(buf++))));
			blocks[4] = (sbuf + (BLOCK_SIZE * (*(buf++))));

			// New sha256
			//block = (uint8_t**)blocks;
			memcpy(data, &s->counter, 8);
			memcpy(db1, blocks[0], BLOCK_SIZE);
			memcpy(db2, blocks[1], BLOCK_SIZE);
			memcpy(db3, blocks[2], BLOCK_SIZE);
			memcpy(db4, blocks[3], BLOCK_SIZE);
			memcpy(db5, blocks[4], BLOCK_SIZE);
			sha256_168byte(data, blocks[1]);
			s->counter++;
		}
		for (size_t i = 1; i < n_blocks; i++) {
			blocks[0] = blocks[1];
			blocks[1] += BLOCK_SIZE;
			blocks[2] = (sbuf + (BLOCK_SIZE * (*(buf++))));
			blocks[3] = (sbuf + (BLOCK_SIZE * (*(buf++))));
			blocks[4] = (sbuf + (BLOCK_SIZE * (*(buf++))));

			// New sha256
			block = (uint8_t**)blocks;
			memcpy(data, &s->counter, 8);
			memcpy(db1, *block++, BLOCK_SIZE);
			memcpy(db2, *block++, BLOCK_SIZE);
			memcpy(db3, *block++, BLOCK_SIZE);
			memcpy(db4, *block++, BLOCK_SIZE);
			memcpy(db5, *block++, BLOCK_SIZE);
			sha256_168byte(data, blocks[1]);
			s->counter++;
		}
		s->has_mixed = true;
	}
}

int hash_state_extract (const struct hash_state *s, uint8_t out[BLOCK_SIZE]) {
  uint8_t *b = block_last (s);
  memcpy ((char *)out, (const char *)b, BLOCK_SIZE);
}

void balloon_reset() {
	prebuf_filled = 0;
}


void balloon_128_orig (unsigned char *input, unsigned char *output) {
  balloon_orig (input, output, 80, 128, 4);
}
void balloon_128_openssl (unsigned char *input, unsigned char *output) {
  balloon_openssl (input, output, 80, 128, 4);
}
void balloon_orig(unsigned char *input, unsigned char *output, int32_t len, int64_t s_cost, int32_t t_cost) {
  struct balloon_options opts;
  struct hash_state s;
  struct timeval tv1,tv2,tv3,tv4,tv5,tv6;
  gettimeofday(&tv1, NULL);
  balloon_init (&opts, s_cost, t_cost);
  gettimeofday(&tv2, NULL);
  hash_state_init (&s, &opts, input);
  gettimeofday(&tv3, NULL);
  hash_state_fill (&s, input, len);
  gettimeofday(&tv4, NULL);
  hash_state_mix_orig(&s, t_cost);
  gettimeofday(&tv5, NULL);
  hash_state_extract (&s, output);
  gettimeofday(&tv6, NULL);
  hash_state_free (&s);
  double init = (double)(tv2.tv_usec - tv1.tv_usec) / 1000000 + (double)(tv2.tv_sec - tv1.tv_sec);
  double state_init = (double)(tv3.tv_usec - tv2.tv_usec) / 1000000 + (double)(tv3.tv_sec - tv2.tv_sec);
  double state_fill = (double)(tv4.tv_usec - tv3.tv_usec) / 1000000 + (double)(tv4.tv_sec - tv3.tv_sec);
  double state_mix = (double)(tv5.tv_usec - tv4.tv_usec) / 1000000 + (double)(tv5.tv_sec - tv4.tv_sec);
  double state_extract = (double)(tv6.tv_usec - tv5.tv_usec) / 1000000 + (double)(tv6.tv_sec - tv5.tv_sec);
  static int printcnt = 0;
  /*printf("Orig    timing: init: %.8f, state_init: %.8f, state_fill: %.8f, state_mix: %.8f, state_extract: %.8f\n",
		  init, state_init, state_fill, state_mix, state_extract);*/
}

void balloon_openssl(unsigned char *input, unsigned char *output, int32_t len, int64_t s_cost, int32_t t_cost) {
  struct balloon_options opts;
  struct hash_state s;
  struct timeval tv1,tv2,tv3,tv4,tv5,tv6;
  gettimeofday(&tv1, NULL);
  balloon_init (&opts, s_cost, t_cost);
  gettimeofday(&tv2, NULL);
  hash_state_init (&s, &opts, input);
  gettimeofday(&tv3, NULL);
  hash_state_fill (&s, input, len);
  gettimeofday(&tv4, NULL);
  hash_state_mix_openssl(&s, t_cost);
  gettimeofday(&tv5, NULL);
  hash_state_extract (&s, output);
  gettimeofday(&tv6, NULL);
  hash_state_free (&s);
  double init = (double)(tv2.tv_usec - tv1.tv_usec) / 1000000 + (double)(tv2.tv_sec - tv1.tv_sec);
  double state_init = (double)(tv3.tv_usec - tv2.tv_usec) / 1000000 + (double)(tv3.tv_sec - tv2.tv_sec);
  double state_fill = (double)(tv4.tv_usec - tv3.tv_usec) / 1000000 + (double)(tv4.tv_sec - tv3.tv_sec);
  double state_mix = (double)(tv5.tv_usec - tv4.tv_usec) / 1000000 + (double)(tv5.tv_sec - tv4.tv_sec);
  double state_extract = (double)(tv6.tv_usec - tv5.tv_usec) / 1000000 + (double)(tv6.tv_sec - tv5.tv_sec);
  static int printcnt = 0;
  printf("OPenSSL timing: init: %.8f, state_init: %.8f, state_fill: %.8f, state_mix: %.8f, state_extract: %.8f\n",
		  init, state_init, state_fill, state_mix, state_extract);
}


void hash_state_mix_orig(struct hash_state *s, int32_t mixrounds) {
	if (!prebuf_filled) {
		bitstream_fill_buffer (&s->bstream, prebuf, 409600);
		prebuf_filled = 1;
		/*printf("buf: ");
		for (int i = 0; i < 8; i++) {
			printf("%x ", prebuf[i]);
		}
		printf("\n");*/
		uint8_t *buf = prebuf;
		uint64_t *lebuf = prebuf_le;
		for (int i = 0; i < 409600; i+=8) {
			bytes_to_littleend8_uint64(buf, lebuf++);
			buf += 8;
		}
		//printf("lebuf: %lx\n", prebuf_le[2]);
	}
	uint64_t *buf = prebuf_le;
	uint64_t neighbor;
	int32_t n_blocks = s->n_blocks;
	//printf("n_blocks: %d\n", n_blocks);
	for (int32_t rounds=0; rounds < mixrounds; rounds++) {
#ifdef TIMED
		struct timeval tv1, tv2;
		gettimeofday(&tv1, NULL);
#endif
		for (size_t i = 0; i < n_blocks; i++) {
			uint8_t *cur_block = (s->buffer + (BLOCK_SIZE * i ));
			const uint8_t *blocks[5];
			*(blocks + 0) = i ? cur_block - BLOCK_SIZE : block_last (s);
			*(blocks + 1) = cur_block;
			*(blocks + 2) = (s->buffer + (BLOCK_SIZE * (*(buf++) % n_blocks)));
			*(blocks + 3) = (s->buffer + (BLOCK_SIZE * (*(buf++) % n_blocks)));
			*(blocks + 4) = (s->buffer + (BLOCK_SIZE * (*(buf++) % n_blocks)));
			compress (&s->counter, cur_block, blocks, 5);
		}
#ifdef TIMED
		gettimeofday(&tv2, NULL);
		double timeus = (double)(tv2.tv_usec - tv1.tv_usec) / 1000000 + (double)(tv2.tv_sec - tv1.tv_sec);
		printf("Inner loop: %.5f\n", timeus);
#endif
		s->has_mixed = true;
	}
}

void hash_state_mix_openssl(struct hash_state *s, int32_t mixrounds) {
	if (!prebuf_filled) {
		bitstream_fill_buffer (&s->bstream, prebuf, 409600);
		prebuf_filled = 1;
		printf("buf: ");
		for (int i = 0; i < 8; i++) {
			printf("%x ", prebuf[i]);
		}
		printf("\n");
		uint8_t *buf = prebuf;
		uint64_t *lebuf = prebuf_le;
		for (int i = 0; i < 409600; i+=8) {
			bytes_to_littleend8_uint64(buf, lebuf++);
			buf += 8;
		}
		printf("lebuf: %lx\n", prebuf_le[2]);
	}
	uint64_t *buf = prebuf_le;
	uint8_t *sbuf = s->buffer;
	uint64_t neighbor;
	int32_t n_blocks = s->n_blocks;
	//printf("n_blocks: %d\n", n_blocks);
	uint8_t *last_block = (sbuf + (BLOCK_SIZE*(n_blocks-1)));
	unsigned char data[8 + BLOCK_SIZE*5];
	const unsigned char *db1 = data+8;
	const unsigned char *db2 = data+40;
	const unsigned char *db3 = data+72;
	const unsigned char *db4 = data+104;
	const unsigned char *db5 = data+136;
	const uint8_t *blocks[5];
	uint8_t **block = (uint8_t**)blocks;
	for (int32_t rounds=0; rounds < mixrounds; rounds++) {
#ifdef TIMED
		struct timeval tv1, tv2;
		gettimeofday(&tv1, NULL);
#endif
		{ // i = 0
			blocks[0] = last_block;
			blocks[1] = sbuf;
			blocks[2] = (sbuf + (BLOCK_SIZE * (*(buf++))));
			blocks[3] = (sbuf + (BLOCK_SIZE * (*(buf++))));
			blocks[4] = (sbuf + (BLOCK_SIZE * (*(buf++))));

			// New sha256
			//block = (uint8_t**)blocks;
#if 0
			memcpy(data, &s->counter, 8);
			memcpy(db1, blocks[0], BLOCK_SIZE);
			memcpy(db2, blocks[1], BLOCK_SIZE);
			memcpy(db3, blocks[2], BLOCK_SIZE);
			memcpy(db4, blocks[3], BLOCK_SIZE);
			memcpy(db5, blocks[4], BLOCK_SIZE);
			//sha256_168byte(data, blocks[1]);
			//s->counter++;
#endif
			compress (&s->counter, blocks[1], blocks, 5);
		}
		for (size_t i = 1; i < n_blocks; i++) {
			blocks[0] = blocks[1];
			blocks[1] += BLOCK_SIZE;
			blocks[2] = (sbuf + (BLOCK_SIZE * (*(buf++))));
			blocks[3] = (sbuf + (BLOCK_SIZE * (*(buf++))));
			blocks[4] = (sbuf + (BLOCK_SIZE * (*(buf++))));
			// New sha256
#if 0
			block = (uint8_t**)blocks;
			memcpy(data, &s->counter, 8);
			memcpy(db1, *block++, BLOCK_SIZE);
			memcpy(db2, *block++, BLOCK_SIZE);
			memcpy(db3, *block++, BLOCK_SIZE);
			memcpy(db4, *block++, BLOCK_SIZE);
			memcpy(db5, *block++, BLOCK_SIZE);
			//sha256_168byte(data, blocks[1]);
			//s->counter++;
#endif
			compress (&s->counter, blocks[1], blocks, 5);
		}
#ifdef TIMED
		gettimeofday(&tv2, NULL);
		double timeus = (double)(tv2.tv_usec - tv1.tv_usec) / 1000000 + (double)(tv2.tv_sec - tv1.tv_sec);
		printf("Inner loop: %.5f\n", timeus);
#endif
		s->has_mixed = true;
	}
}


#ifdef __cplusplus
}
#endif
