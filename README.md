## cuballoon
_cuda miner supporting balloon128/4_

### credits
cpuminer-multi, authored by tpruvot

modified for balloon 128/4, authored by barrystyle

optimized balloon, authored by Belgarion ( accepting donations at: (deft) dJP7aS2GVbmSKbHrS9aRFycdab5UNd4zxa )

Cuda conversion, authored by Belgarion

### installation
 * apt update
 * apt install build-essential autoconf automake libssl-dev libcurl4-openssl-dev libjansson-dev zlib1g-dev screen git cuda-9-2 nvidia-396
 * git clone https://github.com/belgarion/cuballoon
 * cd cuballoon
 * ./build.sh

### Requirements
* Latest Nvidia drivers.

### Tuning
Tune by modifying --cuda_threads and --cuda_blocks.
Good starting points:
GTX1060: --cuda_threads 64 --cuda_blocks 48
GTX1080: --cuda_threads 384 --cuda_blocks 48
GTX1080Ti: --cuda_threads 448 --cuda_blocks 48

### Known bugs
* A few invalid shares when devfee kicks in.
* Sometimes not all GPUs are hashing (verify with taskmgr or nvidia-smi that they have compute utilization).

### Other info
Devfee is around 3%.
