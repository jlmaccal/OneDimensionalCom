#ifndef OPENMM_CUDAEXAMPLEKERNELFACTORY_H_
#define OPENMM_CUDAEXAMPLEKERNELFACTORY_H_

#include "openmm/KernelFactory.h"

namespace OpenMM {

/**
 * This KernelFactory creates kernels for the CUDA implementation of the OneDimComplugin.
 */

class CudaOneDimComKernelFactory : public KernelFactory {
public:
    KernelImpl* createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const;
};

} // namespace OpenMM

#endif /*OPENMM_CUDAEXAMPLEKERNELFACTORY_H_*/
