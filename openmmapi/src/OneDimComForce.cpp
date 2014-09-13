#include "OneDimComForce.h"
#include "internal/OneDimComForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include <vector>
#include <algorithm>
#include <math.h>


using namespace OneDimComPlugin;
using namespace OpenMM;
using namespace std;

OneDimComForce::OneDimComForce(const vector<int>& group1, const vector<int>& group2,
        const vector<float>& weights1, const vector<float>& weights2,
        float k, float r0):
        group1(group1), group2(group2), weights1(weights1),
        weights2(weights2), k(k), r0(r0) {
    validate();
}

const vector<int>& OneDimComForce::getGroup1Indices() const {
    return group1;
}

const vector<int>& OneDimComForce::getGroup2Indices() const {
    return group2;
}

const vector<float>& OneDimComForce::getGroup1Weights() const {
    return weights1;
}

const vector<float>& OneDimComForce::getGroup2Weights() const {
    return weights1;
}

float OneDimComForce::getForceConst() const {
    return k;
}

float OneDimComForce::getR0() const {
    return r0;
}


void OneDimComForce::setGroup1Indices(const vector<int>& indices) {
    if(indices.size() != group1.size()) {
        throw OpenMMException("Size does not match when setting group1.");
    }
    group1 = indices;
    validate();
}

void OneDimComForce::setGroup2Indices(const vector<int>& indices) {
    if(indices.size() != group2.size()) {
        throw OpenMMException("Size does not match when setting group2.");
    }
    group2 = indices;
    validate();
}

void OneDimComForce::setGroup1Weights(const vector<float>& weights) {
    if(weights.size() != weights1.size()) {
        throw OpenMMException("Size does not match when setting weights1.");
    }
    weights1 = weights;
    validate();
}

void OneDimComForce::setGroup2Weights(const vector<float>& weights) {
    if(weights.size() != weights2.size()) {
        throw OpenMMException("Size does not match when setting weights2.");
    }
    weights2 = weights;
    validate();
}

void OneDimComForce::setForceConst(float new_k) {
    k = new_k;
}

void OneDimComForce::setR0(float new_r0) {
    r0 = new_r0;
}

void OneDimComForce::validate() {
    if(group1.size() != weights1.size()) {
        throw OpenMMException("group1 and weights1 are not the same length");
    }

    if(group2.size() != weights2.size()) {
        throw OpenMMException("group2 and weights2 are not the same length");
    }

    float total = 0.0;
    for(std::vector<float>::iterator it=weights1.begin(); it!=weights1.end(); ++it) {
        if(*it < 0.0) {
            throw OpenMMException("weights1 contains value < 0.");
        }
        if(*it > 1.0) {
            throw OpenMMException("weights1 contains value > 0.");
        }
        total += *it;
    }
    if(fabs(total - 1.0) > 1.0e-4) {
        throw OpenMMException("weights1 does not sum to 1.0");
    }

    total = 0.0;
    for(std::vector<float>::iterator it=weights2.begin(); it!=weights2.end(); ++it) {
        if(*it < 0.0) {
            throw OpenMMException("weights2 contains value < 0.");
        }
        if(*it > 1.0) {
            throw OpenMMException("weights2 contains value > 0.");
        }
        total += *it;
    }
    if(fabs(total - 1.0) > 1.0e-4) {
        throw OpenMMException("weights2 does not sum to 1.0");
    }
}

ForceImpl* OneDimComForce::createImpl() const {
    return new OneDimComForceImpl(*this);
}

void OneDimComForce::updateParametersInContext(Context& context) {
    validate();
    dynamic_cast<OneDimComForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}
