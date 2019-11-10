#include <thrust/complex.h>

//GTX1070
#define SM				1 //1 .. 15
#define CUDA_PER_SM		128
#define THREAD_PER_SM 	256 //32X - maximium 1024, 256 is optimal number

#define FOR() for(int i = blockIdx.x*blockDim.x + threadIdx.x; !(i>>n) && i < (1<<n); i += blockDim.x * gridDim.x) 

using cudouble = thrust::complex<double>;

__global__ void cuda_prepare_state(cudouble *data, int n, int period, cudouble amp) {
	FOR() {
		data[i] = (i %  (period) == 0 ? (amp) : 0.0);
	}
}

void gpu_prepare_state(int sm, cudouble *data, int n, int period) {
	const int total_period = ((1 << n) - 1) / period + 1;
	const cudouble amp = 1.0 / sqrt(total_period);

	cuda_prepare_state<<<sm, THREAD_PER_SM>>>(data, n, period, amp);
}

__global__ void cuda_hadamard(cudouble *data, int n, const cudouble sqrt_1_2, int mask_q) {
	FOR() {
		if (i & mask_q) continue;
		const int ii = i ^ mask_q;
		const cudouble a = sqrt_1_2 * (data[i] + data[ii]);
		const cudouble b = sqrt_1_2 * (data[i] - data[ii]);
		data[i] = a;
		data[ii] = b;
	}
}

void gpu_hadamard(int sm, cudouble *data, int n, int q) {
	static const cudouble sqrt_1_2 = sqrt(0.5);
	const int mask_q = 1 << q;
	cuda_hadamard<<<sm, THREAD_PER_SM>>>(data, n, sqrt_1_2, mask_q);
}

__global__ void cuda_controlled_rz(cudouble *data, int n, const cudouble omega, const int mask_q) {
	FOR() {
		if ((~i) & mask_q) continue;
		data[i] *= omega;
	}
}

void gpu_controlled_rz(int sm, cudouble *data, int n, const cudouble omega, const int mask_q) {
	cuda_controlled_rz<<<sm, THREAD_PER_SM>>>(data, n, omega, mask_q);
}

void gpu_init(cudouble **data, int n) {
	cudaMalloc(data, sizeof(cudouble) * (1 << n));
}

void gpu_deinit(cudouble *data) {
	cudaFree(data);
}

void gpu_memcpy(cudouble *dst, cudouble *src, int n) {
	cudaMemcpy(dst, src, sizeof(cudouble) * (1 << n), cudaMemcpyDeviceToHost);
}
