#ifndef BOB_VISIONER_CV_DETECTOR_H
#define BOB_VISIONER_CV_DETECTOR_H

#include "visioner/model/model.h"
#include "visioner/util/geom.h"

namespace bob { namespace visioner {

  // Detection: score + region (at the original scale) + label index
  typedef std::pair<scalar_t, std::pair<rect_t, int> >	detection_t;
  typedef std::vector<detection_t>                        detections_t;
  typedef detections_t::const_iterator                    detections_const_it;
  typedef detections_t::iterator                          detections_it;

  inline detection_t make_detection(scalar_t score, const rect_t& reg, int ilabel)
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
        index_t         m_gts;          // #ground truth objects
        index_t         m_sws;          // #SWs processed (in total)
        index_t         m_evals;        // #LUT evaluations (in total)
        scalar_t        m_timing;       // total 
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
      CVDetector(const std::string& model, scalar_t threshold=0.0,
          index_t levels=0, index_t scale_variation=2, scalar_t clustering=0.05,
          Type detection_method=GroundTruth);

      // Load an image (build the image pyramid)
      bool load(const string_t& ifile, const string_t& gfile);
      bool load(const ipscale_t& ipscale);
      bool load(const grey_t* image, index_t rows, index_t cols);

      // Detect objects
      // NB: The detections are thresholded and clustered!
      bool scan(detections_t& detections) const;

      // Label detections
      bool label(const detection_t& detection) const;
      void label(const detections_t& detections, bools_t& labels) const;
      void label(const detections_t& detections, bool_mat_t& labels) const;

      // Match detections with ground truth locations
      bool match(const detection_t& detection, Object& object) const;

      // Prune detections (remove false alarms)
      void prune(detections_t& detections) const;

      // Compute the ROC - the number of true positives and false alarms
      void evaluate(const strings_t& ifiles, const strings_t& gfiles,
          scalars_t& fas, scalars_t& tars);

      // Check the validity of different components
      bool valid() const;
      bool valid_model() const;
      bool valid_pyramid() const;	
      bool valid_output(index_t output) const;

      // Access functions
      const param_t& param() const { return m_model->param(); }
      const ipyramid_t& ipyramid() const { return m_ipyramid; }
      const ipscale_t& ipscale() const { return m_ipyramid[0]; }
      const objects_t& objects() const { return ipscale().m_objects; }
      index_t n_objects() const { return objects().size(); }
      index_t n_outputs() const { return m_model->n_outputs(); }
      int find(const Object& obj) const { return param().find(obj.type()); }
      const stats_t& stats() const { return m_stats; }
      static scalar_t MinOverlap() { return 0.50; }	

      // Getters and setters
      void set_scan_levels(index_t levels);
      index_t get_scan_levels() const { return m_levels; }

      // Process detections
      static void sort_asc(detections_t& detections);
      static void sort_desc(detections_t& detections);

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

      static void threshold(detections_t& detections, scalar_t thres);
      static void cluster(detections_t& detections, scalar_t thres, index_t n_outputs);                 

      // Compute the ROC - the number of true positives and false alarms
      //	for the <min_score + t * delta_score, t < n_thress> threshold values.
      static void roc(const bool_mat_t& labels, const detections_t& detections,
          scalar_t min_score, index_t n_thress, scalar_t delta_score,
          indices_t& n_tps, indices_t& n_fas);

    public: //attributes

      index_t  m_ds;        ///< Scanning resolution
      scalar_t m_cluster;	  ///< NMS threshold
      scalar_t m_threshold;	///< Detection threshold
      Type     m_type;      ///< Mode: scanning vs. GT

    private: //attributes

      rmodel_t    m_model;	       ///< Object classifier(s)
      index_mat_t m_lmodel_begins; ///< Level classifiers for each output:
      index_mat_t m_lmodel_ends;   ///< [begin, end) LUT range
      index_t			m_levels;	       ///< #levels (speed-up scanning)
      ipyramid_t  m_ipyramid;	     ///< Pyramid of images
      mutable stats_t m_stats;     ///< Scanning statistics

  };

}}

#endif // BOB_VISIONER_CV_DETECTOR_H
