/**
 * @file visioner/visioner/cv/cv_localizer.h
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

#ifndef BOB_VISIONER_CV_LOCALIZER_H
#define BOB_VISIONER_CV_LOCALIZER_H

#include "bob/visioner/cv/cv_detector.h"
#include "bob/visioner/util/histogram.h"

namespace bob { namespace visioner {

  /**
   * Keypoint localizer initialized with either ground truth or valid object
   * detections. The method CVLocalizer::locate() predicts the location of the
   * keypoints. The method CVLocalizer::evaluate() computes the error histogram
   * for each keypoint and cumulated.
   */
  class CVLocalizer {

    public:

      // Localization methods:
      // - one shot (only the detection) vs. multiple shots (collect
      //   predictions nearby the detection)
      // - take the average or the mean of the predictions nearby
      enum Type
      {
        SingleShot = 0,
        MultipleShots_Average,
        MultipleShots_Median
      };

      /**
       * Constructor
       *
       * @param model file containing the localization model to be loaded
       * @param method SingleShot, MultipleShots_Average, MultipleShots_Median
       */
      CVLocalizer(const std::string& model, Type method=MultipleShots_Median);

      // Predict the location of the keypoints in the <reg> region.
      bool locate(const CVDetector& detector,
          const QRectF& reg, std::vector<QPointF>& points) const;

      // Compute the normalized distances [0.0 - 1.0] between
      //	each ground truth keypoints and its predicted points.
      void evaluate(const std::vector<std::string>& ifiles, const std::vector<std::string>& gfiles,
          CVDetector& detector, std::vector<Histogram>& histos, Histogram&
          histo) const;

      // Compute the normalized distances [0.0 - 1.0] between
      //	each ground truth keypoints and its predicted points.
      void evaluate(const std::vector<std::string>& gfiles, const std::vector<std::string>& pfiles,
          std::vector<Histogram>& histos, Histogram& histo) const;

      // Check the validity of different components
      bool valid() const;
      bool valid_model() const;

      // Access functions
      const param_t& param() const { return m_model->param(); }

      // Save the model back to file
      void save(const std::string& filename) const;

    public: //allows command line processing

      /**
       * Default constructor
       */
      CVLocalizer();

      /**
       * Adds options to the parser
       */
      void add_options(boost::program_options::options_description& po_desc) const;

      /**
       * Decodes command line options
       */
      bool decode(const boost::program_options::options_description& po_desc,
          boost::program_options::variables_map& po_vm);

    private:

      // Compute the normalized distances [0.0 - 1.0] between
      //	each ground truth keypoint and its predicted points <dt_points>
      bool evaluate(	
          const Object& gt_object, const std::vector<QPointF>& dt_points,
          std::vector<Histogram>& histos, Histogram& histo) const;

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

      // Number of keypoints
      uint64_t n_points() const { return param().m_labels.size(); }

    public:

      // Attributes
      boost::shared_ptr<Model>                m_model;	// Keypoint localizers
      Type			m_type;
  };

}}

#endif // BOB_VISIONER_CV_LOCALIZER_H
