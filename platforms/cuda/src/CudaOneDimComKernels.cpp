#include "CudaOneDimComKernels.h"
#include "CudaOneDimComKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/cuda/CudaBondedUtilities.h"
#include "openmm/cuda/CudaForceInfo.h"

using namespace OneDimComPlugin;
using namespace OpenMM;
using namespace std;

CudaCalcOneDimComForceKernel::CudaCalcOneDimComForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::CudaContext& cu, const OpenMM::System& system) :
            CalcOneDimComForceKernel(name, platform), hasInitializedKernel(false), cu(cu), system(system), indices(NULL), weights(NULL), h_indices(0), h_weights(0),
            forceConst(0.0), r0(0.0)
{
    if (cu.getUseDoublePrecision()) {
        cout << "***\n";
        cout << "*** MeldForce does not support double precision.\n";
        cout << "***" << endl;
        throw OpenMMException("MeldForce does not support double precision");
    }
}

CudaCalcOneDimComForceKernel::~CudaCalcOneDimComForceKernel() {
    cu.setAsCurrent();
    if (indices != NULL) {
        delete indices;
        indices = NULL;
    }
    if (weights != NULL) {
        delete weights;
        indices = NULL;
    }
}

void CudaCalcOneDimComForceKernel::setupIndicesAndWeights(const OneDimComForce& force) {
    // concatenate the indices into a single vector
    h_indices.clear();
    h_indices.reserve(force.getGroup1Indices().size() + force.getGroup2Indices().size());
    h_indices.insert(h_indices.end(), force.getGroup1Indices().begin(), force.getGroup1Indices().end());
    h_indices.insert(h_indices.end(), force.getGroup2Indices().begin(), force.getGroup2Indices().end());


    // concatenate the weights, negating weights2
    h_weights.clear();
    h_weights.reserve(force.getGroup1Weights().size() + force.getGroup2Weights().size());
    vector<float> w2_copy(force.getGroup2Weights().size());
    copy(force.getGroup2Weights().begin(), force.getGroup2Weights().end(), w2_copy.begin());
    for(vector<float>::iterator it=w2_copy.begin(); it!=w2_copy.end(); ++it) {
        *it = -*it;
    }
    h_weights.insert(h_weights.end(), force.getGroup1Weights().begin(), force.getGroup1Weights().end());
    h_weights.insert(h_weights.end(), w2_copy.begin(), w2_copy.end());
}

void CudaCalcOneDimComForceKernel::initialize(const System& system, const OneDimComForce& force) {
    cu.setAsCurrent();

    setupIndicesAndWeights(force);
    forceConst = force.getForceConst();
    r0 = force.getR0();

    numAtoms = force.getGroup1Indices().size() + force.getGroup2Indices().size();
    if (numAtoms == 0)
        return;

    indices = CudaArray::create<int>(cu, numAtoms, "indices");
    weights = CudaArray::create<float>(cu, numAtoms, "weights");

    indices->upload(h_indices);
    weights->upload(h_weights);

    map<string, string> replacements;
    map<string, string> defines;
    defines["NUM_ATOMS"] = cu.intToString(cu.getNumAtoms());
    defines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
    CUmodule module = cu.createModule(cu.replaceStrings(CudaOneDimComKernelSources::vectorOps + CudaOneDimComKernelSources::computeOneDimComForce, replacements), defines);
    computeForceKernel = cu.getKernel(module, "computeOneDimComForce");
}

double CudaCalcOneDimComForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    void* args[] = {
        &cu.getPosq().getDevicePointer(),
        &numAtoms,
        &forceConst,
        &r0,
        &indices->getDevicePointer(),
        &weights->getDevicePointer(),
        &cu.getForce().getDevicePointer(),
        &cu.getEnergyBuffer().getDevicePointer() };

    // we run with a fixed thread count and block size to ensure
    // that we always run this kernel as a single thread block.
    // All devices of compute capability 2.0 or higher support
    // 1024 threads in a single thread block
    cu.executeKernel(computeForceKernel, args, 1024, 1024, 1024 * sizeof(float));
    return 0.0;
}

void CudaCalcOneDimComForceKernel::copyParametersToContext(ContextImpl& context, const OneDimComForce& force) {
    cu.setAsCurrent();
    setupIndicesAndWeights(force);
    forceConst = force.getForceConst();
    r0 = force.getR0();

    indices->upload(h_indices);
    weights->upload(h_weights);

    cu.invalidateMolecules();
}
