#ifndef BOB_VISIONER_CV_LOCALIZER_H
#define BOB_VISIONER_CV_LOCALIZER_H

#include "visioner/cv/cv_detector.h"
#include "visioner/util/histogram.h"

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
          const rect_t& reg, points_t& points) const;

      // Compute the normalized distances [0.0 - 1.0] between
      //	each ground truth keypoints and its predicted points.
      void evaluate(const strings_t& ifiles, const strings_t& gfiles,
          CVDetector& detector, std::vector<Histogram>& histos, Histogram&
          histo) const;

      // Compute the normalized distances [0.0 - 1.0] between
      //	each ground truth keypoints and its predicted points.
      void evaluate(const strings_t& gfiles, const strings_t& pfiles,
          std::vector<Histogram>& histos, Histogram& histo) const;

      // Check the validity of different components
      bool valid() const;
      bool valid_model() const;

      // Access functions
      const param_t& param() const { return m_model->param(); }

      // Save the model back to file
      void save(const std::string& filename) const;

    private:

      // Compute the normalized distances [0.0 - 1.0] between
      //	each ground truth keypoint and its predicted points <dt_points>
      bool evaluate(	
          const Object& gt_object, const points_t& dt_points,
          std::vector<Histogram>& histos, Histogram& histo) const;

      // Collect predictions from the neighbourhood of <seed_sw>
      void locate(const CVDetector& detector, const sw_t& seed_sw, 
          int n_ds, int n_dx, int dx, int n_dy, int dy,
          std::vector<points_t>& preds) const;

      // Average the collection of predictions
      void avg(const std::vector<points_t>& preds, points_t& pred) const;
      void avg(const points_t& pts, point_t& pt) const;

      // Median the collection of predictions
      void med(const std::vector<points_t>& preds, points_t& pred) const;
      void med(const points_t& pts, point_t& pt) const;

      // Number of keypoints
      index_t n_points() const { return param().m_labels.size(); }

    public:

      // Attributes
      rmodel_t                m_model;	// Keypoint localizers
      Type			m_type;
  };

}}

#endif // BOB_VISIONER_CV_LOCALIZER_H
