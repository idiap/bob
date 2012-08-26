/**
 * @file visioner/visioner/cv/cv_detector.h
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

#ifndef BOB_VISIONER_CV_DETECTOR_H
#define BOB_VISIONER_CV_DETECTOR_H

#include "visioner/model/model.h"
#include "visioner/util/geom.h"

namespace bob { namespace visioner {

  // Detection: score + region (at the original scale) + label index
  typedef std::pair<double, std::pair<QRectF, int> >	detection_t;

  inline detection_t make_detection(double score, const QRectF& reg, int ilabel)
  {
    return std::make_pair(score, std::make_pair(reg, ilabel));
  }

  /////////////////////////////////////////////////////////////////////////////////////////
  // Object detector that processes a pyramid of images:
  //	::scan()	-> return the object detections (thresholded & clustered)
  //	::label()	-> label detections as TP/FA
  //	::evaluate()	-> computes TPs/FAs ROC curve
  //
  // NB: It can be used to detect multiple object types.
  /////////////////////////////////////////////////////////////////////////////////////////

  class CVDetector
  {
    public:

      // Runtime statistics
      struct stats_t
      {
        // Constructor
        stats_t()
          :       m_gts(0), m_sws(0), m_evals(0), m_timing(0.0)
        {                                
        }

        // Display the statistics
        void show() const;

        // Attributes
        uint64_t         m_gts;          // #ground truth objects
        uint64_t         m_sws;          // #SWs processed (in total)
        uint64_t         m_evals;        // #LUT evaluations (in total)
        double        m_timing;       // total 
      };

      enum Type
      {
        Scanning,
        GroundTruth
      };

      /**
       * Constructor from scratch
       *
       * @param model file containing the model to be loaded
       * @param threshold object classification threshold
       * @param levels levels (the more, the faster)
       * @param scale_variation scale variation in pixels
       * @param clustering overlapping threshold for clustering detections
       * @param detection_method Scanning or GroundTruth
       */
      CVDetector(const std::string& model, double threshold=0.0,
          uint64_t levels=0, uint64_t scale_variation=2, double clustering=0.05,
          Type detection_method=GroundTruth);

      // Load an image (build the image pyramid)
      bool load(const std::string& ifile, const std::string& gfile);
      bool load(const ipscale_t& ipscale);
      bool load(const uint8_t* image, uint64_t rows, uint64_t cols);

      // Detect objects
      // NB: The detections are thresholded and clustered!
      bool scan(std::vector<detection_t>& detections) const;

      // Label detections
      bool label(const detection_t& detection) const;
      void label(const std::vector<detection_t>& detections, std::vector<int>& labels) const;
      void label(const std::vector<detection_t>& detections, Matrix<int>& labels) const;

      // Match detections with ground truth locations
      bool match(const detection_t& detection, Object& object) const;

      // Prune detections (remove false alarms)
      void prune(std::vector<detection_t>& detections) const;

      // Compute the ROC - the number of true positives and false alarms
      void evaluate(const std::vector<std::string>& ifiles, const std::vector<std::string>& gfiles,
          std::vector<double>& fas, std::vector<double>& tars);

      // Check the validity of different components
      bool valid() const;
      bool valid_model() const;
      bool valid_pyramid() const;	
      bool valid_output(uint64_t output) const;

      // Access functions
      const param_t& param() const { return m_model->param(); }
      const ipyramid_t& ipyramid() const { return m_ipyramid; }
      const ipscale_t& ipscale() const { return m_ipyramid[0]; }
      const std::vector<Object>& objects() const { return ipscale().m_objects; }
      uint64_t n_objects() const { return objects().size(); }
      uint64_t n_outputs() const { return m_model->n_outputs(); }
      int find(const Object& obj) const { return param().find(obj.type()); }
      const stats_t& stats() const { return m_stats; }
      static double MinOverlap() { return 0.50; }	

      // Getters and setters
      void set_scan_levels(uint64_t levels);
      uint64_t get_scan_levels() const { return m_levels; }

      // Process detections
      static void sort_asc(std::vector<detection_t>& detections);
      static void sort_desc(std::vector<detection_t>& detections);

      // Save the model back to file
      void save(const std::string& filename) const;

    public: //allows command line processing

      /**
       * Default constructor
       */
      CVDetector();

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

      static void threshold(std::vector<detection_t>& detections, double thres);
      static void cluster(std::vector<detection_t>& detections, double thres, uint64_t n_outputs);                 

      // Compute the ROC - the number of true positives and false alarms
      //	for the <min_score + t * delta_score, t < n_thress> threshold values.
      static void roc(const Matrix<int>& labels, const std::vector<detection_t>& detections,
          double min_score, uint64_t n_thress, double delta_score,
          std::vector<uint64_t>& n_tps, std::vector<uint64_t>& n_fas);

    public: //attributes

      uint64_t  m_ds;        ///< Scanning resolution
      double m_cluster;	  ///< NMS threshold
      double m_threshold;	///< Detection threshold
      Type     m_type;      ///< Mode: scanning vs. GT

    private: //attributes

      boost::shared_ptr<Model>    m_model;	       ///< Object classifier(s)
      Matrix<uint64_t> m_lmodel_begins; ///< Level classifiers for each output:
      Matrix<uint64_t> m_lmodel_ends;   ///< [begin, end) LUT range
      uint64_t			m_levels;	       ///< number of levels (speed-up scanning)
      ipyramid_t  m_ipyramid;	     ///< Pyramid of images
      mutable stats_t m_stats;     ///< Scanning statistics

  };

}}

#endif // BOB_VISIONER_CV_DETECTOR_H
