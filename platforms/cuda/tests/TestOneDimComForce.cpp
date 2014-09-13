#include "OneDimComForce.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/Context.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/OpenMMException.h"
#include <cmath>
#include <iostream>
#include <vector>

using namespace OneDimComPlugin;
using namespace OpenMM;
using namespace std;

extern "C" OPENMM_EXPORT void registerOneDimComCudaKernelFactories();

void testTwoParticles() {
    System system;
    vector<Vec3> positions(3);

    // three particles, but the middle one is not included
    // int the force in order to catch stupid indexing errors
    system.addParticle(1.0);
    system.addParticle(1.0);
    system.addParticle(1.0);
    positions[0] = Vec3(1.0, 0.0, 0.0);
    positions[1] = Vec3(200.0, 0.0, 0.0);
    positions[2] = Vec3(2.0, 0.0, 0.0);

    vector<int> group1;
    vector<int> group2;
    vector<float> weights1;
    vector<float> weights2;
    group1.push_back(0);
    group2.push_back(2);
    weights1.push_back(1.0);
    weights2.push_back(1.0);

    OneDimComForce* force = new OneDimComForce(group1, group2, weights1, weights2, 1.0, 2.0);
    system.addForce(force);

    VerletIntegrator integrator(1.0);
    Platform& platform = Platform::getPlatformByName("CUDA");
    Context context(system, integrator, platform);
    context.setPositions(positions);

    State state = context.getState(State::Energy | State::Forces);

    // check energy
    ASSERT_EQUAL_TOL(0.5, state.getPotentialEnergy(), 1e-5);

    // check the forces
    float expectedForce = 1.0;
    ASSERT_EQUAL_TOL(-expectedForce, state.getForces()[0][0], 1e-5);
    ASSERT_EQUAL_TOL(expectedForce, state.getForces()[2][0], 1e-5);
}

void testManyParticles() {
    // test with a large number of particles to ensure that
    // things work when the number of particles is larger
    // than the block size
    System system;
    const int numParticlesPerGroup = 5000;
    vector<Vec3> positions(numParticlesPerGroup * 2);
    vector<int> group1, group2;
    vector<float> weights1, weights2;

    for (int i=0; i<numParticlesPerGroup; ++i) {
        system.addParticle(1.0);
        positions[i] = Vec3(1.0, 0.0, 0.0);
        group1.push_back(i);
        weights1.push_back(1.0 / numParticlesPerGroup);
    }

    for (int i=numParticlesPerGroup; i<(2 * numParticlesPerGroup); ++i) {
        system.addParticle(1.0);
        positions[i] = Vec3(2.0, 0.0, 0.0);
        group2.push_back(i);
        weights2.push_back(1.0 / numParticlesPerGroup);
    }

    OneDimComForce* force = new OneDimComForce(group1, group2, weights1, weights2, 1.0, 2.0);
    system.addForce(force);

    VerletIntegrator integrator(1.0);
    Platform& platform = Platform::getPlatformByName("CUDA");
    Context context(system, integrator, platform);
    context.setPositions(positions);

    State state = context.getState(State::Energy | State::Forces);

    // check energy
    ASSERT_EQUAL_TOL(0.5, state.getPotentialEnergy(), 1e-5);

    // check the forces
    float expectedForce = 1.0 / numParticlesPerGroup;
    ASSERT_EQUAL_TOL(-expectedForce, state.getForces()[0][0], 1e-5);
    ASSERT_EQUAL_TOL(expectedForce, state.getForces()[numParticlesPerGroup][0], 1e-5);
}

void testChangingParameters() {
    System system;
    vector<Vec3> positions(3);

    // three particles, but the middle one is not included
    // int the force in order to catch stupid indexing errors
    system.addParticle(1.0);
    system.addParticle(1.0);
    system.addParticle(1.0);
    positions[0] = Vec3(1.0, 0.0, 0.0);
    positions[1] = Vec3(200.0, 0.0, 0.0);
    positions[2] = Vec3(2.0, 0.0, 0.0);

    vector<int> group1;
    vector<int> group2;
    vector<float> weights1;
    vector<float> weights2;
    group1.push_back(0);
    group2.push_back(2);
    weights1.push_back(1.0);
    weights2.push_back(1.0);

    OneDimComForce* force = new OneDimComForce(group1, group2, weights1, weights2, 1.0, 2.0);
    system.addForce(force);

    VerletIntegrator integrator(1.0);
    Platform& platform = Platform::getPlatformByName("CUDA");
    Context context(system, integrator, platform);
    context.setPositions(positions);

    State state = context.getState(State::Energy | State::Forces);

    // now change the parameters
    // flip group 1 and 2
    force->setGroup1Indices(group2);
    force->setGroup2Indices(group1);
    // double the force constant
    force->setForceConst(2.0);
    // flip R0 to the other direction
    force->setR0(-2.0);
    // push the changes to the gpu
    force->updateParametersInContext(context);

    // check energy
    state = context.getState(State::Energy | State::Forces);
    ASSERT_EQUAL_TOL(1.0, state.getPotentialEnergy(), 1e-5);

    // check the forces
    float expectedForce = 2.0;
    ASSERT_EQUAL_TOL(-expectedForce, state.getForces()[0][0], 1e-5);
    ASSERT_EQUAL_TOL(expectedForce, state.getForces()[2][0], 1e-5);
}

void testGroupSum1() {
    // Create a OneDimComForce where the group 1 weights don't add up to one
    int g1[] = {0, 1};
    int g2[] = {2, 3};
    float w1[] = {0.5, 0.0};
    float w2[] = {0.5, 0.5};

    std::vector<int> group1(g1, g1 + sizeof(g1) / sizeof(g1[0]));
    std::vector<int> group2(g2, g2 + sizeof(g2) / sizeof(g2[0]));
    std::vector<float> weights1(w1, w1 + sizeof(w1) / sizeof(w1[0]));
    std::vector<float> weights2(w2, w2 + sizeof(w2) / sizeof(w2[0]));
    float k = 1.0;
    float r0 = 1.0;

    try {
        OneDimComForce* force = new OneDimComForce(group1, group2, weights1, weights2, k, r0);
    }
    catch (OpenMMException e) {
        // we're supposed to throw an exception, so we return successfully if we get here.
        return;
    }
    // we shouldn't get here
    throw OpenMMException("Should have thrown an exception when weights1 didn't sum to 1.0");
}

void testGroupSum2() {
    // Create a OneDimComForce where the group 2 weights don't add up to one
    int g1[] = {0, 1};
    int g2[] = {2, 3};
    float w1[] = {0.5, 0.5};
    float w2[] = {0.5, 0.0};

    std::vector<int> group1(g1, g1 + sizeof(g1) / sizeof(g1[0]));
    std::vector<int> group2(g2, g2 + sizeof(g2) / sizeof(g2[0]));
    std::vector<float> weights1(w1, w1 + sizeof(w1) / sizeof(w1[0]));
    std::vector<float> weights2(w2, w2 + sizeof(w2) / sizeof(w2[0]));
    float k = 1.0;
    float r0 = 1.0;

    try {
        OneDimComForce* force = new OneDimComForce(group1, group2, weights1, weights2, k, r0);
    }
    catch (OpenMMException e) {
        // we're supposed to throw an exception, so we return successfully if we get here.
        return;
    }
    // we shouldn't get here
    throw OpenMMException("Should have thrown an exception when weights2 didn't sum to 1.0");
}

void testSizeMatch1() {
    // Create a OneDimComForce where the size of group1 and weights 1 don't match
    int g1[] = {0, 1};
    int g2[] = {2, 3};
    float w1[] = {0.25, 0.25, 0.25, 0.25};
    float w2[] = {0.5, 0.5};

    std::vector<int> group1(g1, g1 + sizeof(g1) / sizeof(g1[0]));
    std::vector<int> group2(g2, g2 + sizeof(g2) / sizeof(g2[0]));
    std::vector<float> weights1(w1, w1 + sizeof(w1) / sizeof(w1[0]));
    std::vector<float> weights2(w2, w2 + sizeof(w2) / sizeof(w2[0]));
    float k = 1.0;
    float r0 = 1.0;

    try {
        OneDimComForce* force = new OneDimComForce(group1, group2, weights1, weights2, k, r0);
    }
    catch (OpenMMException e) {
        // we're supposed to throw an exception, so we return successfully if we get here.
        return;
    }
    // we shouldn't get here
    throw OpenMMException("Should have thrown an exception when group1 and weights1 have different sizes.");
}

void testSizeMatch2() {
    // Create a OneDimComForce where the size of group2 and weights 2 don't match
    int g1[] = {0, 1};
    int g2[] = {2, 3};
    float w1[] = {0.5, 0.5};
    float w2[] = {0.25, 0.25, 0.25, 0.25};

    std::vector<int> group1(g1, g1 + sizeof(g1) / sizeof(g1[0]));
    std::vector<int> group2(g2, g2 + sizeof(g2) / sizeof(g2[0]));
    std::vector<float> weights1(w1, w1 + sizeof(w1) / sizeof(w1[0]));
    std::vector<float> weights2(w2, w2 + sizeof(w2) / sizeof(w2[0]));
    float k = 1.0;
    float r0 = 1.0;

    try {
        OneDimComForce* force = new OneDimComForce(group1, group2, weights1, weights2, k, r0);
    }
    catch (OpenMMException e) {
        // we're supposed to throw an exception, so we return successfully if we get here.
        return;
    }
    // we shouldn't get here
    throw OpenMMException("Should have thrown an exception when group2 and weights2 have different sizes.");
}

int main(int argc, char* argv[]) {
    try {
        registerOneDimComCudaKernelFactories();
        if (argc > 1)
            Platform::getPlatformByName("CUDA").setPropertyDefaultValue("CudaPrecision", string(argv[1]));

        // run the tests
        testGroupSum1();
        testGroupSum2();
        testSizeMatch1();
        testSizeMatch2();
        testTwoParticles();
        testManyParticles();
        testChangingParameters();

        /* testForce(); */
        /* testChangingParameters(); */
    }
    catch(const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}
