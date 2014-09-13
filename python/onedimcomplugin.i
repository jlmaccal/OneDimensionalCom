%module onedimcomplugin

%import(module="simtk.openmm") "OpenMMSwigHeaders.i"


/*
 * The following lines are needed to handle std::vector.
 * Similar lines may be needed for vectors of vectors or
 * for other STL types like maps.
 */

%include "std_vector.i"
namespace std {
  %template(vectorf) vector<float>;
  %template(vectori) vector<int>;
};

%{
#include "OneDimComForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
%}


/*
 * The code below strips all units before the wrapper
 * functions are called. This code also converts numpy
 * arrays to lists.
*/

%pythoncode %{
import simtk.openmm as mm
import simtk.unit as unit
%}


/* strip the units off of all input arguments */
%pythonprepend %{
try:
    args=mm.stripUnits(args)
except UnboundLocalError:
    pass
%}


/*
 * Add units to function outputs.
*/
%pythonappend OneDimComPlugin::OneDimComForce::getForceConst() const %{
    val[0] = unit.Quantity(val[0], unit.kilojoule_per_mole / (unit.nanometer * unit.nanometer))
%}

%pythonappend OneDimComPlugin::OneDimComForce::getR0() const %{
    val[0] = unit.Quantity(val[0], unit.nanometer)
%}

namespace OneDimComPlugin {

class OneDimComForce : public OpenMM::Force {
public:
    OneDimComForce(const std::vector<int>& group1,
                   const std::vector<int>& group2,
                   const std::vector<float>& weights1,
                   const std::vector<float>& weights2,
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
};

}

