/**
 * @file visioner/src/cv_detector.cc
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

#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/format.hpp>

#include "core/logging.h"

#include "visioner/cv/cv_detector.h"
#include "visioner/model/mdecoder.h"
#include "visioner/util/timer.h"

namespace bob { namespace visioner {

  // Constructor
  CVDetector::CVDetector():	
    m_ds(2),
    m_cluster(0.05),
    m_threshold(0.0),
    m_type(GroundTruth),
    m_levels(0)
  {
  }

  // Command line processing
  template <typename T>
    void decode_var(const boost::program_options::options_description& po_desc,
        boost::program_options::variables_map& po_vm,
        const char* var_name, T& var)
    {
      if (!po_vm.count(var_name))
      {
        bob::core::error << po_desc << std::endl;
        exit(EXIT_FAILURE);
      }

      var = po_vm[var_name].as<T>();
    }

  void CVDetector::add_options(boost::program_options::options_description& po_desc) const
  {
    po_desc.add_options()

      ("detect_model", boost::program_options::value<std::string>(),
       "detection: object classification model")

      ("detect_threshold",
       boost::program_options::value<scalar_t>()->default_value(m_threshold),
       "detection: object classification threshold")

      ("detect_levels", 
       boost::program_options::value<index_t>()->default_value(m_levels),
       "detection: levels (the more, the faster)")

      ("detect_ds",
       boost::program_options::value<index_t>()->default_value(m_ds),
       "detection: scale variation in pixels")
      
      ("detect_cluster",
       boost::program_options::value<scalar_t>()->default_value(m_cluster),
       "detection: overlapping threshold for clustering detections")
      
      ("detect_method",
       boost::program_options::value<std::string>()->default_value("groundtruth"),
       "detection: method (scanning, groundtruth)");

  }

  bool CVDetector::decode(const boost::program_options::options_description& po_desc, boost::program_options::variables_map& po_vm) {

    // Load the model
    const std::string cmd_model = po_vm["detect_model"].as<std::string>();
    if (Model::load(cmd_model, m_model) == false)
    {
      bob::core::error 
        << "Failed to load the model <" << cmd_model << ">!" << std::endl;
      return false;
    }
    if (valid_model() == false)
    {
      bob::core::error << "Invalid model!" << std::endl;
      return false;
    }

    param_t _param = param();
    _param.m_ds = m_ds;
    m_ipyramid.reset(_param); 

    // Decode parameters
    decode_var(po_desc, po_vm, "detect_threshold", m_threshold);
    decode_var(po_desc, po_vm, "detect_levels", m_levels);
    decode_var(po_desc, po_vm, "detect_ds", m_ds);
    decode_var(po_desc, po_vm, "detect_cluster", m_cluster);     

    string_t cmd_method;
    decode_var(po_desc, po_vm, "detect_method", cmd_method);

    if (cmd_method == "groundtruth")
    {
      m_type = GroundTruth;
    }
    else if (cmd_method == "scanning")
    {
      m_type = Scanning;                         
    }
    else
    {
      bob::core::error << "Invalid detection method!" << std::endl;
      return false;
    }

    set_scan_levels(m_levels);

    // OK
    return true;
  }

  CVDetector::CVDetector(const std::string& model, scalar_t threshold,
      index_t levels, index_t scale_variation, scalar_t clustering,
      CVDetector::Type detection_method):
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

      set_scan_levels(levels);

    }



  void CVDetector::set_scan_levels(index_t levels) {
    m_levels = levels;

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
        bob::core::warn << "Failed to load image <" << ifile << "> or ground truth <" << gfile << ">!" << std::endl;
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
      bob::core::info 
        << "Image [" << (i + 1) << "/" << ifiles.size() << "]: produced "
        << detections.size() << " detections / "
        << n_objects() << " GTs in " << timer.elapsed() << "s." << std::endl;
    }

    const index_t n_thress = 256;
    const scalar_t delta_score = inverse(n_thress - 1) * (max_score - min_score);

    bob::core::info 
      << "min_score = " << min_score << ", max_score = " << max_score
      << ", delta_score = " << delta_score << std::endl;

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
    bob::core::info << "Processed " << m_gts << " GTs by scanning " 
      << m_sws << " SWs with " << (inverse(m_sws) * m_evals) 
      << " LUT evaluations done in " << (inverse(m_sws) * m_timing) 
      << " seconds on average." << std::endl;
  }

  // Save the model back to file
  void CVDetector::save(const std::string& filename) const
  {
    m_model->save(filename);
  }

}}
