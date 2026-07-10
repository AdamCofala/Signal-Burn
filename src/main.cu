#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>

struct IQSample {
    int16_t r;
    int16_t i;
};

__global__ void convertInt16ToComplex(const IQSample* in, cufftComplex* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx].x = (float)in[idx].r;
        out[idx].y = (float)in[idx].i;
    }
}

__global__ void calcMagnitude(const cufftComplex* in, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float real = in[idx].x;
        float imag = in[idx].y;
        out[idx] = sqrtf(real * real + imag * imag);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) return 1;

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    size_t size = std::stoull(argv[3]);

    std::vector<IQSample> buffer(size);
    FILE* in_file = (input_path == "-") ? stdin : fopen(input_path.c_str(), "rb");
    if (!in_file) return 1;
    fread(buffer.data(), sizeof(IQSample), size, in_file);
    if (input_path != "-") fclose(in_file);

    IQSample* d_in;
    cufftComplex* d_out;
    float* d_mag;

    cudaMalloc((void**)&d_in, size * sizeof(IQSample));
    cudaMalloc((void**)&d_out, size * sizeof(cufftComplex));
    cudaMalloc((void**)&d_mag, size * sizeof(float));

    cudaMemcpy(d_in, buffer.data(), size * sizeof(IQSample), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    convertInt16ToComplex<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, size);
    cudaDeviceSynchronize();

    cufftHandle plan;
    cufftPlan1d(&plan, size, CUFFT_C2C, 1);
    cufftExecC2C(plan, d_out, d_out, CUFFT_FORWARD);
    cudaDeviceSynchronize();

    calcMagnitude<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_mag, size);
    cudaDeviceSynchronize();

    std::vector<float> h_mag(size);
    cudaMemcpy(h_mag.data(), d_mag, size * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* out = fopen(output_path.c_str(), "wb");
    if (out) {
        fwrite(h_mag.data(), sizeof(float), size, out);
        fclose(out);
    }

    cufftDestroy(plan);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_mag);

    return 0;
}