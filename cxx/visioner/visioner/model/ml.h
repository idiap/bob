#ifndef BOB_VISIONER_ML_H
#define BOB_VISIONER_ML_H

#include "visioner/util/util.h"

namespace bob { namespace visioner {

  // Optimization type
  enum OptimizationType
  {
    Expectation,    // Expectation loss formulation
    Variational     // Variational loss formulation
  };

  // Feature sharing method
  enum FeatureSharingType
  {
    Independent,    // A feature for each output
    Shared          // Single feature for all outputs
  };

  /////////////////////////////////////////////////////////////////////////////////////////
  // Labelling convention for classification
  /////////////////////////////////////////////////////////////////////////////////////////

  inline scalar_t pos_target() { return 1.0; }
  inline scalar_t neg_target() { return -1.0; }

  /////////////////////////////////////////////////////////////////////////////////////////
  // Compute the classification/regression error for some
  //      targets and predicted scores
  /////////////////////////////////////////////////////////////////////////////////////////

  scalar_t classification_error(scalar_t target, scalar_t score, scalar_t epsilon);
  scalar_t regression_error(scalar_t target, scalar_t score, scalar_t epsilon);        

  /////////////////////////////////////////////////////////////////////////////////////////
  // ROC processing
  /////////////////////////////////////////////////////////////////////////////////////////

  // Compute the area under the ROC curve
  scalar_t roc_area(const scalars_t& fars, const scalars_t& tars);

  // Order the FARs and TARs such that to make a real curve
  void roc_order(scalars_t& fars, scalars_t& tars);	

  // Trim the ROC curve (remove points that line inside a horizontal segment)
  void roc_trim(scalars_t& fars, scalars_t& tars);

  // Save the ROC points to file
  bool save_roc(const scalars_t& fars, const scalars_t& tars, const string_t& path);

}}

#endif // BOB_VISIONER_ML_H
