/**
 * @file visioner/visioner/cv/cv_classifier.h
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

#ifndef BOB_VISIONER_CV_CLASSIFIER_H
#define BOB_VISIONER_CV_CLASSIFIER_H

#include "bob/visioner/cv/cv_detector.h"

namespace bob { namespace visioner {

  /////////////////////////////////////////////////////////////////////////////////////////
  // Object classification initialized with either ground truth or valid object detections:
  //	::classify()	-> predict the object type/pose/id
  //	::evaluate()	-> computes the confusion matrix or the 
  /////////////////////////////////////////////////////////////////////////////////////////

  class CVClassifier
  {
    public:

      // Constructor
      CVClassifier();

      // Command line processing
      void add_options(boost::program_options::options_description& po_desc) const;
      bool decode(	const boost::program_options::options_description& po_desc,
          boost::program_options::variables_map& po_vm);

      // Predict the object label of the keypoints in the <reg> region.
      bool classify(	const CVDetector& detector, const QRectF& reg, uint64_t& dt_label) const;

      // Compute the confusion matrix considering the 
      //      ground truth and the predicted labels.
      void evaluate(  const std::vector<std::string>& ifiles, const std::vector<std::string>& gfiles,
          CVDetector& detector, 
          Matrix<uint64_t>& hits_mat, std::vector<uint64_t>& hits_cnt) const;

      // Retrieve the ground truth label for the given object
      bool classify(  const Object& object, uint64_t& gt_label) const;

      // Check the validity of different components
      bool valid() const;
      bool valid_model() const;

      // Access functions
      const param_t& param() const { return m_model->param(); }
      uint64_t n_classes() const { return param().m_labels.size(); }
      const std::string& label(uint64_t i) const { return param().m_labels[i]; }

    private:

      // Collect predictions from the neighbourhood of <seed_sw>
      void locate(const CVDetector& detector, const subwindow_t& seed_sw, 
          int n_ds, int n_dx, int dx, int n_dy, int dy,
          std::vector<std::vector<QPointF> >& preds) const;

      // Average the collection of predictions
      void avg(const std::vector<std::vector<QPointF> >& preds, std::vector<QPointF>& pred) const;
      void avg(const std::vector<QPointF>& pts, QPointF& pt) const;

      // Median the collection of predictions
      void med(const std::vector<std::vector<QPointF> >& preds, std::vector<QPointF>& pred) const;
      void med(const std::vector<QPointF>& pts, QPointF& pt) const;

      // Return the neighbours (location only) of the given sub-window
      std::vector<subwindow_t> neighbours(const CVDetector& detector, const subwindow_t& sw, 
          int n_dx, int dx, int n_dy, int dy) const;

      // Number of keypoints
      uint64_t n_points() const { return param().m_labels.size(); }

    private:

      // Attributes
      boost::shared_ptr<Model>                m_model;	// Object classifiers
  };

}}

#endif // BOB_VISIONER_CV_CLASSIFIER_H
