#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/format.hpp>

#include "visioner/cv/cv_detector.h"
#include "visioner/model/mdecoder.h"
#include "visioner/util/timer.h"

namespace bob { namespace visioner {

  CVDetector::CVDetector(const std::string& model, scalar_t threshold,
      index_t levels, index_t scale_variation, scalar_t clustering,
      CVDetector::Type detection_method):
    m_levels(levels),
    m_ds(scale_variation),
    m_cluster(clustering),
    m_threshold(threshold),
    m_type(detection_method) {

    // Load the model
    if (Model::load(model, m_model) == false) {
      boost::format m("failed to load model from file '%s'");
      m % model;
      throw std::runtime_error(m.str().c_str());
    }

    if (valid_model() == false) {
      boost::format m("the model loaded from file '%s' is not valid");
      m % model;
      throw std::runtime_error(m.str().c_str());
    }

    param_t _param = param();
    _param.m_ds = m_ds;
    m_ipyramid.reset(_param); 

    // Build the level classifiers
    m_lmodel_begins.resize(n_outputs(), m_levels + 1);
    m_lmodel_ends.resize(n_outputs(), m_levels + 1);
    for (index_t o = 0; o < n_outputs(); o ++)
    {
      const index_t size = m_model->n_luts(o);

      m_lmodel_begins(o, 0) = 0;
      m_lmodel_ends(o, 0) = size >> m_levels;

      for (index_t l = 1; l <= m_levels; l ++)			
      {
        m_lmodel_begins(o, l) = size >> (m_levels - l + 1); 
        m_lmodel_ends(o, l) = size >> (m_levels - l);
      }
    }
  }

  // Load an image (build the image pyramid)
  bool CVDetector::load(const string_t& ifile, const string_t& gfile)
  {
    return	m_ipyramid.load(ifile, gfile) &&
      m_ipyramid.empty() == false;
  }
  bool CVDetector::load(const ipscale_t& ipscale)
  {
    return	m_ipyramid.load(ipscale) &&
      m_ipyramid.empty() == false;
  }
  bool CVDetector::load(const grey_t* image, index_t rows, index_t cols)
  {
    return	m_ipyramid.load(image, rows, cols) &&
      m_ipyramid.empty() == false;
  }

  // Check the validity of different components
  bool CVDetector::valid() const
  {
    return  valid_model() && valid_pyramid();
  }
  bool CVDetector::valid_model() const
  {
    return	m_model.get() != 0 &&
      m_model->n_outputs() == param().m_labels.size();
  }
  bool CVDetector::valid_pyramid() const
  {
    return  m_ipyramid.empty() == false;
  }
  bool CVDetector::valid_output(index_t output) const
  {
    return	output >= 0 && output < m_model->n_outputs();
  }

  // Detect objects
  // NB: The detections are thresholded and clustered!
  bool CVDetector::scan(detections_t& detections) const
  {
    detections.clear();   

    // Ground truth mode ...
    if (m_type == GroundTruth)
    {
      for (objects_t::const_iterator it = objects().begin(); it != objects().end(); ++ it)
      {
        const int ilabel = find(*it);
        if (ilabel >= 0)
        {
          detections.push_back(make_detection(0.0, it->bbx(), ilabel));

          // Update statistics
          m_stats.m_gts ++;
        }
      }
      return true;
    }

    // Scanning mode ...

    // Check parameters
    if (valid_model() == false)
    {
      return false;
    }

    // Scan the image ... 
    Timer timer;
    for (index_t is = 0; is < m_ipyramid.size(); is ++)
    {
      const ipscale_t& ip = m_ipyramid[is];
      m_model->preprocess(ip);

      // ... with every model type
      for (index_t o = 0; o < n_outputs(); o ++)
      {
        for (int x = ip.m_scan_min_x; x < ip.m_scan_max_x; x += ip.m_scan_dx)
          for (int y = ip.m_scan_min_y; y < ip.m_scan_max_y; y += ip.m_scan_dy)
          {
            // Concentrate computation on the most promising detections
            scalar_t score = 0.0;
            for (index_t l = 0; l <= m_levels && score >= 0.0; l ++)
            {
              const index_t lbegin = m_lmodel_begins[o][l];
              const index_t lend = m_lmodel_ends[o][l];
              score += m_model->score(o, lbegin, lend, x, y);

              // Update statistics
              m_stats.m_evals += lend - lbegin;
            }

            // Threshold detection and map it to the original image size
            if (score >= m_threshold)
            {
              detections.push_back(make_detection(
                    score, 
                    m_ipyramid.map(sw_t(x, y, is)), 
                    o));
            }

            // Update statistics
            m_stats.m_sws ++;
          }
      }
    }

    // Update statistics
    m_stats.m_gts += n_objects();
    m_stats.m_timing += timer.elapsed();

    // OK, cluster detections
    cluster(detections, m_cluster, n_outputs());
    return true;
  }

  // Match detections with ground truth locations
  bool CVDetector::match(const detection_t& detection, Object& object) const
  {
    int which = 0;
    if (    overlap(detection.second.first, objects(), &which) >= MinOverlap() &&
        detection.second.second == find(objects()[which]))
    {
      object = objects()[which];
      return true;
    }
    else
    {
      return false;
    }
  }

  // Label detections
  bool CVDetector::label(const detection_t& detection) const
  {
    int which = 0;
    return  overlap(detection.second.first, objects(), &which) >= MinOverlap() &&
      detection.second.second == find(objects()[which]);
  }
  void CVDetector::label(const detections_t& detections, bools_t& labels) const
  {
    labels.resize(detections.size());
    for (index_t id = 0; id < detections.size(); id ++)
    {
      labels[id] = label(detections[id]);
    }
  }
  void CVDetector::label(const detections_t& detections, bool_mat_t& labels) const
  {
    labels.resize(detections.size(), n_objects());
    for (index_t id = 0; id < detections.size(); id ++)
    {
      const detection_t& det = detections[id];
      for (index_t ig = 0; ig < n_objects(); ig ++)
      {
        const Object& obj = objects()[ig];
        labels(id, ig) = 
          overlap(det.second.first, obj.bbx()) >= MinOverlap() &&
          det.second.second == find(obj);
      }
    }
  }

  // Prune detections (remove false alarms)
  void CVDetector::prune(detections_t& detections) const
  {
    Object object;
    for (index_t i = 0; i < detections.size(); i ++)
    {
      if (match(detections[i], object) == false)
      {
        detections.erase(detections.begin() + i);
        i --;
      }
    }
  }

  // Process detections
  void CVDetector::sort_asc(detections_t& detections)
  {
    std::sort(detections.begin(), detections.end(), std::less<detection_t>());
  }

  void CVDetector::sort_desc(detections_t& detections)
  {
    std::sort(detections.begin(), detections.end(), std::greater<detection_t>());
  }

  void CVDetector::threshold(detections_t& detections, scalar_t thres)
  {
    detections.erase(
        std::remove_if(	detections.begin(), detections.end(),
          boost::lambda::bind(&detection_t::first, boost::lambda::_1) < thres),
        detections.end());
  }

  void CVDetector::cluster(detections_t& detections, scalar_t thres, index_t n_outputs)
  {
    if (thres >= 1.0)
    {
      // Clustering deactivated!
      return;
    }

    sort_desc(detections);

    // Check if the detection ...
    detections_t result;
    for (index_t o = 0; o < n_outputs; o ++)
    {
      for (index_t iref = 0; iref < detections.size(); iref ++)
      {
        const detection_t& ref = detections[iref];
        if (    ref.second.first.left() > -0.5 && // ... is still valid!
            ref.second.second == (int)o)
        {
          result.push_back(ref);

          // Then, remove the overlapping detections
          for (index_t icrt = iref + 1; icrt < detections.size(); icrt ++)
          {
            detection_t& crt = detections[icrt];
            if (	crt.second.first.left() > -0.5 &&
                crt.second.second == (int)o &&
                overlap(ref.second.first, crt.second.first) >= thres)
            {
              crt.second.first.setLeft(-1.0);
            }
          }
        }
      }
    }

    detections.swap(result);
  }

  // Compute the ROC - the number of true positives and false alarms
  //	for the <min_score + t * delta_score, t < n_thress> threshold values.
  void CVDetector::roc(
      const bool_mat_t& labels, const detections_t& detections,
      scalar_t min_score, index_t n_thress, scalar_t delta_score,
      indices_t& n_tps, indices_t& n_fas)
  {
    if (	detections.empty() ||
        labels.rows() != detections.size())
    {
      return;
    }

    // Compute the FAs and TPs for various threshold values
    scalar_t thres = min_score;
    for (index_t t = 0, id = 0; t < n_thress; t ++, thres += delta_score)
    {
      // Find the first detection with the score just above this threshold
      for ( ; id < detections.size() &&
          detections[id].first <= thres; id ++)
      {
      }

      // Update the number of false alarms and true detections
      //	for the thresholded detections
      index_t tps = 0, fas = 0;

      for (index_t cid = id; cid < labels.rows(); cid ++)
      {
        fas ++;
        for (index_t ig = 0; ig < labels.cols(); ig ++)
        {
          if (labels(cid, ig))
          {
            fas --;
            break;
          }
        }
      }

      for (index_t ig = 0; ig < labels.cols(); ig ++)
      {
        for (index_t cid = id; cid < labels.rows(); cid ++)
        {
          if (labels(cid, ig))
          {
            tps ++;
            break;
          }
        }
      }

      n_fas[t] += fas;
      n_tps[t] += tps;
    }
  }

  // Compute the ROC - the number of true positives and false alarms
  void CVDetector::evaluate(
      const strings_t& ifiles, const strings_t& gfiles,
      scalars_t& fas, scalars_t& tars)
  {
    // 1st pass: process each image to assess the minimum/maximum score ...
    scalar_t min_score = std::numeric_limits<scalar_t>::max();
    scalar_t max_score = -std::numeric_limits<scalar_t>::max();

    std::vector<detections_t>       idetections;
    std::vector<bool_mat_t>         ilabels;

    detections_t detections;

    for (index_t i = 0; i < ifiles.size(); i ++)
    {
      const std::string& ifile = ifiles[i];
      const std::string& gfile = gfiles[i];

      // Load the image and the ground truth
      if (load(ifile, gfile) == false)
      {
        log_warning("CVDetector", "evaluate") 
          << "Failed to load image <" << ifile << "> or ground truth <" << gfile << ">!\n";
        continue;
      }

      // Scan the image ...
      Timer timer;

      scan(detections);

      // Update the score extremes
      sort_asc(detections);
      if (detections.empty() == false)
      {
        min_score = std::min(min_score, detections.begin()->first);
        max_score = std::max(max_score, detections.rbegin()->first);
      }

      // Save detections and labels
      bool_mat_t labels;
      label(detections, labels); 

      idetections.push_back(detections);		
      ilabels.push_back(labels);                

      // Debug
      log_info("CVDetector", "evaluate") 
        << "Image [" << (i + 1) << "/" << ifiles.size() << "]: produced "
        << detections.size() << " detections / "
        << n_objects() << " GTs in " << timer.elapsed() << "s.\n";
    }

    const index_t n_thress = 256;
    const scalar_t delta_score = inverse(n_thress - 1) * (max_score - min_score);

    log_info("CVDetector", "evaluate") 
      << "min_score = " << min_score << ", max_score = " << max_score
      << ", delta_score = " << delta_score << "\n";

    std::vector<index_t> n_tps(n_thress, 0);	// #true positives
    std::vector<index_t> n_fas(n_thress, 0);	// #false alarms			

    // 2nd pass: use the detections and the ground truth locations to compute the ROC curve ...
    for (index_t i = 0; i < ifiles.size(); i ++)
    {
      const detections_t& detections = idetections[i];
      const bool_mat_t& labels = ilabels[i];

      roc(	labels, detections, min_score, n_thress, delta_score,
          n_tps, n_fas);
    }

    // Build the ROC curve		
    fas.resize(n_thress), tars.resize(n_thress);
    for (index_t t = 0; t < n_thress; t ++)
    {
      fas[t] = n_fas[t];
      tars[t] = inverse(stats().m_gts) * n_tps[t];
    }			
    roc_order(fas, tars);
    roc_trim(fas, tars);    

    // Display statistics
    stats().show();
  }

  // Display statistics
  void CVDetector::stats_t::show() const
  {
    log_info("CVDetector::stats_t", "show")
      << "Processed " << m_gts << " GTs by scanning " << m_sws << " SWs with " 
      << (inverse(m_sws) * m_evals) << " LUT evaluations done in "
      << (inverse(m_sws) * m_timing) << " seconds on average.\n";
  }

  // Save the model back to file
  void CVDetector::save(const std::string& filename) const
  {
    m_model->save(filename);
  }

}}
