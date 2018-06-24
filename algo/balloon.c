#include "miner.h"
#include <string.h>
#include <stdint.h>
#include <unistd.h>

#include <openssl/sha.h>

#include "balloon/balloon.h"

#define CUDA
//#define DEBUG_VERIFY
//#define DEBUG
//#define PRINT_ENDIANDATA
uint32_t prev_pdata[20][10];
int scanhash_balloon(int thr_id, struct work *work, uint32_t max_nonce, uint64_t *hashes_done, uint32_t num_cuda_threads, uint32_t num_cuda_blocks) {
	uint32_t _ALIGN(128) hash32[8];
	uint32_t _ALIGN(128) orighash32[8];
	uint32_t _ALIGN(128) sslhash32[8];
	uint32_t _ALIGN(128) cudahash32[8];
	uint32_t _ALIGN(128) verifyhash32[8];
	uint32_t _ALIGN(128) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;

	uint8_t pdata_changed = 0;
	for (int i = 0; i < 10; i++) {
		if (prev_pdata[thr_id][i] != pdata[i]) {
			prev_pdata[thr_id][i] = pdata[i];
			pdata_changed = 1;
		}
	}

	/*printf("ptarget: ");
	for (int i = 0; i < 8; i++) {
		printf("%x ", ptarget[i]);
	}
	printf("\n");*/
	//ptarget[7] = 0xffffff;

	const uint32_t Htarg = ptarget[7];
	const uint32_t first_nonce = pdata[19];
	//if (thr_id >= 2) return 0;
	//printf("first_nonce: %x\n", first_nonce);

	uint32_t n = first_nonce;

	for (int i = 0; i < 19; i++) {
		be32enc(&endiandata[i], pdata[i]);
	};

	balloon_cuda_init(thr_id, opt_cuda_syncmode, num_cuda_threads, num_cuda_blocks);
	if (pdata_changed) {
		balloon_reset();
		reset_host_prebuf();
	}
	do {
		be32enc(&endiandata[19], n);
#ifdef PRINT_ENDIANDATA
		if (n % 100 == 0) {
			printf("endiandata: ");
			for (int i = 0; i < 20; i++) {
				printf("%x ", endiandata[i]);
			}
			printf("\n");
		}
#endif
#ifdef CUDA
		uint32_t is_winning = 0;
		uint32_t winning_nonce = balloon_128_cuda(thr_id, (unsigned char *)endiandata, (unsigned char*)cudahash32, ptarget, max_nonce, num_cuda_threads, &is_winning, num_cuda_blocks);
		be32enc(&endiandata[19], winning_nonce);
		n = winning_nonce;
		if (is_winning) {
			balloon_128_orig((unsigned char *)endiandata, (unsigned char*)verifyhash32);
			memcpy(hash32, verifyhash32, 32);

			//memcpy(hash32, cudahash32, 32);
#ifdef DEBUG_VERIFY
			uint8_t verify_successful = 1;
			for (int i = 0; i < 8; i++) {
				if (cudahash32[i] != verifyhash32[i]) {
					printf("WARNING: Hash mismatch!\n");
					verify_successful = 0;
					break;
				}
			}
			if (!verify_successful) {
				printf("cudahash: ");
				for (int i = 0; i < 8; i++) {
					printf("%x ", cudahash32[i]);
				}
				printf("\n");
				printf("verifyhash: ");
				for (int i = 0; i < 8; i++) {
					printf("%x ", verifyhash32[i]);
				}
				printf("\n");
			}
#endif // DEBUG
		}
#else
		balloon_128((unsigned char *)endiandata, (unsigned char *)hash32);
#ifdef DEBUG
		printf("hash: ");
		for (int i = 0; i < 8; i++) {
			printf("%x ", hash32[i]);
		}
		printf("\n");
#endif //DEBUG
#endif

#ifdef DEBUG
		balloon_128_orig((unsigned char *)endiandata, (unsigned char*)orighash32);
		balloon_128_openssl((unsigned char *)endiandata, (unsigned char*)sslhash32);
		//	usleep(500000);
		printf("orighash: ");
		for (int i = 0; i < 8; i++) {
			printf("%x ", orighash32[i]);
		}
		printf("\n");
		printf("sslhash: ");
		for (int i = 0; i < 8; i++) {
			printf("%x ", sslhash32[i]);
		}
		printf("\n");
#endif
		//printf("hash32[7] = %x, Htarg = %x\n", hash32[7], Htarg);
		if (is_winning && hash32[7] < Htarg && fulltest(hash32, ptarget)) {
			printf("Submitting nonce %d on gpu %u\n", winning_nonce, thr_id);
			/*printf("hash: ");
			for (int i = 0; i < 8; i++) {
				printf("%x ", hash32[i]);
			}
			printf("\n");
			printf("pdata: ");
			for (int i = 0; i < 20; i++) {
				printf("%02x ", pdata[i]);
			}
			printf("\n");*/
			work_set_target_ratio(work, hash32);
			*hashes_done = n - first_nonce + 1;
			pdata[19] = n;
			balloon_cuda_free(thr_id);
			return true;
		}
		n++;

	} while (n < max_nonce && !work_restart[thr_id].restart);
	balloon_cuda_free(thr_id);

	*hashes_done = n - first_nonce + 1;
	pdata[19] = n;

	return 0;
}
