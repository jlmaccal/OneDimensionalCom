/* -------------------------------------------------------------------------- *
 *                                OpenMMExample                                 *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2014 Stanford University and the Authors.           *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "OneDimComForce.h"
#include "openmm/Platform.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/serialization/XmlSerializer.h"
#include <iostream>
#include <sstream>
#include <vector>

using namespace OneDimComPlugin;
using namespace OpenMM;
using namespace std;

extern "C" void registerOneDimComSerializationProxies();

void testSerialization() {
    // Create a Force.
    vector<int> g1;
    g1.push_back(0);

    vector<int> g2;
    g2.push_back(1);
    g2.push_back(2);

    vector<float> w1;
    w1.push_back(1.0);

    vector<float> w2;
    w2.push_back(0.5);
    w2.push_back(0.5);

    float k = 2.0;
    float r0 = 1.0;

    // create the force
    OneDimComForce force(g1, g2, w1, w2, k, r0);

    // serialize and then deserialize it
    stringstream buffer;
    XmlSerializer::serialize<OneDimComForce>(&force, "Force", buffer);
    OneDimComForce* copy = XmlSerializer::deserialize<OneDimComForce>(buffer);
    // Compare the two forces to see if they are identical.
    OneDimComForce& force2 = *copy;
    ASSERT_EQUAL(force.getForceConst(), force2.getForceConst());
    ASSERT_EQUAL(force.getR0(), force2.getR0());

    // get all of the groups and weights
    vector<int> g1_orig = force.getGroup1Indices();
    vector<int> g2_orig = force.getGroup2Indices();
    vector<float> w1_orig = force.getGroup1Weights();
    vector<float> w2_orig = force.getGroup2Weights();

    vector<int> g1_copy = force.getGroup1Indices();
    vector<int> g2_copy = force.getGroup2Indices();
    vector<float> w1_copy = force.getGroup1Weights();
    vector<float> w2_copy = force.getGroup2Weights();

    // make sure the lengths match
    ASSERT_EQUAL(g1_orig.size(), g1_copy.size());
    ASSERT_EQUAL(g2_orig.size(), g2_copy.size());
    ASSERT_EQUAL(w1_orig.size(), w1_copy.size());
    ASSERT_EQUAL(w2_orig.size(), w2_copy.size());

    // now make sure all of the indices and weights for group1 match
    for (int i=0; i<g1_orig.size(); ++i) {
        ASSERT_EQUAL(g1_orig[i], g1_copy[i]);
        ASSERT_EQUAL(w1_orig[i], w1_orig[i]);
    }

    // now make sure all of the indices and weights for group1 match
    for (int i=0; i<g2_orig.size(); ++i) {
        ASSERT_EQUAL(g2_orig[i], g2_copy[i]);
        ASSERT_EQUAL(w2_orig[i], w2_orig[i]);
    }
}

int main() {
    try {
        registerOneDimComSerializationProxies();
        testSerialization();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
