#ifndef CUDA_EXAMPLE_KERNELS_H_
#define CUDA_EXAMPLE_KERNELS_H_

#include "OneDimComKernels.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"

namespace OneDimComPlugin {

/**
 * This kernel is invoked by OneDimComForce to calculate the forces acting on the system and the energy of the system.
 */
class CudaCalcOneDimComForceKernel : public CalcOneDimComForceKernel {
public:
    CudaCalcOneDimComForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::CudaContext& cu, const OpenMM::System& system);

    ~CudaCalcOneDimComForceKernel();
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the OneDimComForce this kernel will be used for
     */
    void initialize(const OpenMM::System& system, const OneDimComForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the OneDimComForce to copy the parameters from
     */
    void copyParametersToContext(OpenMM::ContextImpl& context, const OneDimComForce& force);
private:
    CUfunction computeForceKernel;
    void setupIndicesAndWeights(const OneDimComForce& force);
    int numAtoms;
    float forceConst;
    float r0;
    std::vector<int> h_indices;
    OpenMM::CudaArray* indices;
    std::vector<float> h_weights;
    OpenMM::CudaArray* weights;
    bool hasInitializedKernel;
    OpenMM::CudaContext& cu;
    const OpenMM::System& system;
};

} // namespace OneDimComPlugin

#endif /*CUDA_EXAMPLE_KERNELS_H_*/
