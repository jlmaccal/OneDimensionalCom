extern "C" __global__ void computeOneDimComForce(const real4* __restrict__ posq, int nAtoms, float k, float r0,
                                      const int* __restrict__ indices, const float* __restrict__ weights,
                                      unsigned long long* __restrict__ forceBuffer, real* __restrict__ energyBuffer) {
    extern __shared__ float accumulator[];

    // our index
    // this kernel is only run with a single thread block
    int threadIndex = threadIdx.x;

    // zero out the accumulator
    accumulator[threadIndex] = 0.0;

    // each thread adds it's values to the accumulator
    for (int index=threadIndex; index<nAtoms; index+=blockDim.x) {
        // we subtract so that the sign is positive when group2 is to the
        // right of group 1
        accumulator[threadIndex] -= posq[indices[index]].x * weights[index];
    }
    __syncthreads();

    // now do a parallel reduction to get the weighted displacement
    for (unsigned int stride=blockDim.x/2; stride>0; stride>>=1) {
        if (threadIndex < stride) {
            accumulator[threadIndex] += accumulator[threadIndex + stride];
        }
        __syncthreads();
    }

    // compute the energy on thread zero
    if (threadIndex == 0) {
        energyBuffer[0] += 0.5 * k * (accumulator[0] - r0) * (accumulator[0] - r0);
    }
    __syncthreads();

    // compute the forces and store in the buffer
    float factor = k * (accumulator[0] - r0);
    for (int index=threadIndex; index<nAtoms; index+=blockDim.x) {
        float force = factor * weights[index];
        atomicAdd(&forceBuffer[indices[index]], static_cast<unsigned long long>((long long)(force*0x100000000)));
    }
}
