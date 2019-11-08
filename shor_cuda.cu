#include <thrust/complex.h>

using cudouble = thrust::complex<double>;

__global__ void cuda_prepare_state(cudouble *data, int n, int period, cudouble amp) {
	for (int i = 0; !(i >> n); ++i) {
	    data[i] = (i % period == 0 ? amp : 0.0);
	  }
}

void gpu_prepare_state(cudouble *data, int n, int period) {
	const int total_period = ((1 << n) - 1) / period + 1;
	const cudouble amp = 1.0 / sqrt(total_period);

	cuda_prepare_state<<<1,1>>>(data, n, period, amp);
}

__global__ void cuda_hadamard(cudouble *data, int n, const cudouble sqrt_1_2, int mask_q) {
	for (int i = 0; !(i >> n); ++i) {
		if (i & mask_q) continue;
		const int ii = i ^ mask_q;
		const cudouble a = sqrt_1_2 * (data[i] + data[ii]);
		const cudouble b = sqrt_1_2 * (data[i] - data[ii]);
		data[i] = a;
		data[ii] = b;
	}
}

void gpu_hadamard(cudouble *data, int n, int q) {
	static const cudouble sqrt_1_2 = sqrt(0.5);
	const int mask_q = 1 << q;
	cuda_hadamard<<<1,1>>>(data, n, sqrt_1_2, mask_q);
}

__global__ void cuda_controlled_rz(cudouble *data, int n, const cudouble omega, const int mask_q) {
	for (int i = 0; !(i >> n); ++i) {
		if ((~i) & mask_q) continue;
		data[i] *= omega;
	}
}

void gpu_controlled_rz(cudouble *data, int n, const cudouble omega, const int mask_q) {
	cuda_controlled_rz<<<1,1>>>(data, n, omega, mask_q);
}