#include <stdio.h>
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
 
#define N 16
const int blocksize = 4; 
 
__global__ 
void init_random(curandState_t *state) 
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	curand_init(0,idx,0,&state[idx]);
}

__global__ 
void generate_random(curandState_t *state, double *b) 
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < N){
		b[idx] = curand_normal_double(&state[idx]);
	}
}
 
int main()
{

	cudaError_t ierrAsync;
    cudaError_t ierrSync;

	double *b;
	curandState_t *state;
	
	// allocate memory in GPU for random number creation state objects.
	cudaMallocManaged(&state, N * sizeof(curandState_t));
	cudaMallocManaged(&b, N * sizeof(double));
	
	dim3 dimBlock( blocksize, 1 );
	dim3 dimGrid( N/blocksize+1, 1 );
	init_random<<<dimGrid, dimBlock>>>(state);
	ierrSync = cudaGetLastError();
	ierrAsync = cudaDeviceSynchronize();
	if (ierrSync != cudaSuccess) { printf("Sync error: %s\n", cudaGetErrorString(ierrSync)); }
    if (ierrAsync != cudaSuccess) { printf("Async error: %s\n", cudaGetErrorString(ierrAsync)); }

	generate_random<<<dimGrid, dimBlock>>>(state,b);
	ierrSync = cudaGetLastError();
	ierrAsync = cudaDeviceSynchronize();
	if (ierrSync != cudaSuccess) { printf("Sync error: %s\n", cudaGetErrorString(ierrSync)); }
    if (ierrAsync != cudaSuccess) { printf("Async error: %s\n", cudaGetErrorString(ierrAsync)); }

	for (int i=0; i<N; i++){
		std::cout << b[i] << std::endl;
	}

	cudaFree( state );
	cudaFree( b );
	
	return EXIT_SUCCESS;
}
