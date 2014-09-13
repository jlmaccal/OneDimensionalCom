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

#include "OneDimComForceProxy.h"
#include "OneDimComForce.h"
#include "openmm/serialization/SerializationNode.h"
#include <sstream>
#include <vector>
#include <iostream>

using namespace OneDimComPlugin;
using namespace OpenMM;
using namespace std;

OneDimComForceProxy::OneDimComForceProxy() : SerializationProxy("OneDimComForce") {
}

void OneDimComForceProxy::serialize(const void* object, SerializationNode& node) const {
    node.setIntProperty("version", 1);
    const OneDimComForce& force = *reinterpret_cast<const OneDimComForce*>(object);
    node.setDoubleProperty("forceConst", force.getForceConst());
    node.setDoubleProperty("r0", force.getR0());

    SerializationNode& group1 = node.createChildNode("group1");
    for (vector<int>::const_iterator it=force.getGroup1Indices().begin(); it!=force.getGroup1Indices().end(); ++it) {
        group1.createChildNode("index").setIntProperty("index", *it);
    }

    SerializationNode& group2 = node.createChildNode("group2");
    for (vector<int>::const_iterator it=force.getGroup2Indices().begin(); it!=force.getGroup2Indices().end(); ++it) {
        group2.createChildNode("index").setIntProperty("index", *it);
    }

    SerializationNode& weights1 = node.createChildNode("weights1");
    for (vector<float>::const_iterator it=force.getGroup1Weights().begin(); it!=force.getGroup1Weights().end(); ++it) {
        weights1.createChildNode("weight").setDoubleProperty("weight", *it);
    }

    SerializationNode& weights2 = node.createChildNode("weights2");
    for (vector<float>::const_iterator it=force.getGroup2Weights().begin(); it!=force.getGroup2Weights().end(); ++it) {
        weights2.createChildNode("weight").setDoubleProperty("weight", *it);
    }
}

void* OneDimComForceProxy::deserialize(const SerializationNode& node) const {
    if (node.getIntProperty("version") != 1)
        throw OpenMMException("Unsupported version number");
    float forceConst = 0.0;
    float r0 = 0.0;
    vector<int> group1;
    vector<int> group2;
    vector<float> weights1;
    vector<float> weights2;
    try {
        forceConst = node.getDoubleProperty("forceConst");
        r0 = node.getDoubleProperty("r0");

        const SerializationNode& group1Node = node.getChildNode("group1");
        for (vector<SerializationNode>::const_iterator it=group1Node.getChildren().begin(); it!=group1Node.getChildren().end(); ++it) {
            group1.push_back(it->getIntProperty("index"));
        }

        const SerializationNode& group2Node = node.getChildNode("group2");
        for (vector<SerializationNode>::const_iterator it=group2Node.getChildren().begin(); it!=group2Node.getChildren().end(); ++it) {
            group2.push_back(it->getIntProperty("index"));
        }

        const SerializationNode& weights1Node = node.getChildNode("weights1");
        for (vector<SerializationNode>::const_iterator it=weights1Node.getChildren().begin(); it!=weights1Node.getChildren().end(); ++it) {
            weights1.push_back(it->getDoubleProperty("weight"));
        }

        const SerializationNode& weights2Node = node.getChildNode("weights2");
        for (vector<SerializationNode>::const_iterator it=weights2Node.getChildren().begin(); it!=weights2Node.getChildren().end(); ++it) {
            weights2.push_back(it->getDoubleProperty("weight"));
        }
    }
    catch (...) {
        throw;
    }
    OneDimComForce* force = new OneDimComForce(group1, group2, weights1, weights2, forceConst, r0);
    return force;
}
