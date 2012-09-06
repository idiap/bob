/**
 * @file bob/visioner/model/ml.h
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef BOB_VISIONER_ML_H
#define BOB_VISIONER_ML_H

#include "bob/visioner/util/util.h"

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

  inline double pos_target() { return 1.0; }
  inline double neg_target() { return -1.0; }

  /////////////////////////////////////////////////////////////////////////////////////////
  // Compute the classification/regression error for some
  //      targets and predicted scores
  /////////////////////////////////////////////////////////////////////////////////////////

  double classification_error(double target, double score, double epsilon);
  double regression_error(double target, double score, double epsilon);        

  /////////////////////////////////////////////////////////////////////////////////////////
  // ROC processing
  /////////////////////////////////////////////////////////////////////////////////////////

  // Compute the area under the ROC curve
  double roc_area(const std::vector<double>& fars, const std::vector<double>& tars);

  // Order the FARs and TARs such that to make a real curve
  void roc_order(std::vector<double>& fars, std::vector<double>& tars);	

  // Trim the ROC curve (remove points that line inside a horizontal segment)
  void roc_trim(std::vector<double>& fars, std::vector<double>& tars);

  // Save the ROC points to file
  bool save_roc(const std::vector<double>& fars, const std::vector<double>& tars, const std::string& path);

}}

#endif // BOB_VISIONER_ML_H
