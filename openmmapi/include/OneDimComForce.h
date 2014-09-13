#ifndef OPENMM_ONEDIMCIMFORCE_H_
#define OPENMM_ONEDIMCIMFORCE_H_


#include "openmm/Context.h"
#include "openmm/Force.h"
#include <vector>
#include "internal/windowsExportExample.h"

namespace OneDimComPlugin {

/**
 * This class implements a harmonic bond force of the for E = k * (R_AB - r_0)^2,
 * where R_AB is the distance between two groups A and B and r_0 is the
 * equilibrium distance.
 */

class OPENMM_EXPORT_EXAMPLE OneDimComForce : public OpenMM::Force {
public:
    /**
     * Create an OneDimComForce.
     */
    OneDimComForce(const std::vector<int>& group1, const std::vector<int>& group2,
            const std::vector<float>& weights1, const std::vector<float>& weights2,
            float k, float r0);

    const std::vector<int>& getGroup1Indices() const;
    const std::vector<int>& getGroup2Indices() const;
    const std::vector<float>& getGroup1Weights() const;
    const std::vector<float>& getGroup2Weights() const;
    float getForceConst() const;
    float getR0() const;

    void setGroup1Indices(const std::vector<int>& indices);
    void setGroup2Indices(const std::vector<int>& indices);
    void setGroup1Weights(const std::vector<float>& weights);
    void setGroup2Weights(const std::vector<float>& weights);
    void setForceConst(float k);
    void setR0(float r0);

    void updateParametersInContext(OpenMM::Context& context);
    void validate();
protected:
    OpenMM::ForceImpl* createImpl() const;
private:
    std::vector<int> group1;
    std::vector<int> group2;
    std::vector<float> weights1;
    std::vector<float> weights2;
    float k, r0;
};

} // namespace OneDimComPlugin

#endif
