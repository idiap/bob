/**
 * @file visioner/cxx/cv_localizer.cc
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

#include <boost/format.hpp>

#include "bob/core/logging.h"

#include "bob/visioner/cv/cv_localizer.h"
#include "bob/visioner/model/mdecoder.h"
#include "bob/visioner/util/timer.h"

namespace bob { namespace visioner {

  CVLocalizer::CVLocalizer()
    :	m_type(MultipleShots_Median)
  {
  }

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

  void CVLocalizer::add_options(boost::program_options::options_description& po_desc) const {

    po_desc.add_options()

      ("localize_model", boost::program_options::value<std::string>(),
       "localization: keypoint localization model")
      
      ("localize_method", 
       boost::program_options::value<std::string>()->default_value("mshots+med"),
       "localization: method (1shot, mshots+avg, mshots+med");

  }

  bool CVLocalizer::decode(const boost::program_options::options_description& po_desc,
      boost::program_options::variables_map& po_vm)
  {
    // Load the localization model
    const std::string cmd_model = po_vm["localize_model"].as<std::string>();
    if (Model::load(cmd_model, m_model) == false)
    {
      bob::core::error 
        << "Failed to load the localization model <" << cmd_model << ">!" << std::endl;
      return false;
    }
    if (valid_model() == false)
    {
      bob::core::error << "Invalid model!" << std::endl;
      return false;
    }

    // Decode parameters
    std::string cmd_method;
    decode_var(po_desc, po_vm, "localize_method", cmd_method);

    if (cmd_method == "1shot")
    {
      m_type = SingleShot;
    }
    else if (cmd_method == "mshots+avg")
    {
      m_type = MultipleShots_Average;
    }
    else if (cmd_method == "mshots+med")
    {
      m_type = MultipleShots_Median;
    }
    else
    {
      bob::core::error << "Invalid localization method!" << std::endl;
      return false;
    }

    // OK
    return true;
  }

  CVLocalizer::CVLocalizer(const std::string& model, Type method):
    m_type(method) {

    // Load the localization model
    if (Model::load(model, m_model) == false) {
      boost::format m("failed to load localization model from file '%s'");
      m % model;
      throw std::runtime_error(m.str());
    }
    if (valid_model() == false) {
      boost::format m("the model loaded from file '%s' is not valid");
      m % model;
      throw std::runtime_error(m.str());
    }
  }

  // Check the validity of different components
  bool CVLocalizer::valid() const
  {
    return valid_model();
  }
  bool CVLocalizer::valid_model() const
  {
    return	m_model.get() != 0 &&
      m_model->n_outputs() == 2 * param().m_labels.size();
  }

  // Predict the location of the keypoints in the <reg> region.
  bool CVLocalizer::locate(const CVDetector& detector, const QRectF& reg, std::vector<QPointF>& dt_points) const
  {
    // Check the sub-window
    const subwindow_t sw = detector.ipyramid().map(reg, param());
    if (detector.ipyramid().check(sw, param()) == false)
    {
      return false;
    }

    // Collect predictions (not projected)
    std::vector<std::vector<QPointF> > preds(n_points());
    switch (m_type)
    {
      case SingleShot:
        locate(detector, sw, 0, 0, 0, 0, 0, preds);
        break;

      case MultipleShots_Average:
      case MultipleShots_Median:			
      default:
        locate(detector, sw, 
            1, 4, std::max((int)1, (int)(0.5 + 0.02 * param().m_cols)), 
            4, std::max((int)1, (int)(0.5 + 0.02 * param().m_rows)), preds);
        break;
    }

    // Process the predictions: average or median (if required)
    std::vector<QPointF> pred;                
    switch (m_type)
    {
      case SingleShot:
        for (uint64_t i = 0; i < n_points(); i ++)
        {
          pred.push_back(preds[i][0]);
        }
        break;

      case MultipleShots_Average:
        avg(preds, pred);
        break;

      case MultipleShots_Median:
      default:
        med(preds, pred);
        break;
    }

    // OK
    dt_points.insert(dt_points.end(), pred.begin(), pred.end());
    return true;
  }

  // Collect predictions from the neighbourhood of <seed_sw>
  void CVLocalizer::locate(
      const CVDetector& detector, const subwindow_t& seed_sw, 
      int n_ds, int n_dx, int dx, int n_dy, int dy, std::vector<std::vector<QPointF> >& preds) const
  {       
    std::vector<std::vector<subwindow_t> > ssws = detector.ipyramid().neighbours(
        seed_sw, n_ds, n_dx, dx, n_dy, dy, param());

    // Process each scale ...
    for (uint64_t ss = 0; ss < ssws.size(); ss ++)
    {
      const std::vector<subwindow_t>& sws = ssws[ss];
      if (sws.empty() == true)
      {
        continue;
      }

      const ipscale_t& ip = detector.ipyramid()[sws[0].m_s];                                                
      m_model->preprocess(ip);

      // Process each sub-window at this scale ...
      const double inv_scale = ip.m_inv_scale;                          
      for (uint64_t s = 0; s < sws.size(); s ++)
      {
        const subwindow_t& sw = sws[s];

        for (uint64_t i = 0; i < n_points(); i ++)
        {
          const double x = m_model->score(2 * i + 0, sw.m_x, sw.m_y) * param().m_cols;
          const double y = m_model->score(2 * i + 1, sw.m_x, sw.m_y) * param().m_rows;

          preds[i].push_back(QPointF(inv_scale * (sw.m_x + x), 
                inv_scale * (sw.m_y + y)));
        }
      }
    }
  }

  // Average the collection of predictions
  void CVLocalizer::avg(const std::vector<std::vector<QPointF> >& preds, std::vector<QPointF>& pred) const
  {
    pred.resize(n_points());
    for (uint64_t i = 0; i < n_points(); i ++)
    {
      avg(preds[i], pred[i]);
    }
  }
  void CVLocalizer::avg(const std::vector<QPointF>& pts, QPointF& pt) const
  {
    pt = std::accumulate(pts.begin(), pts.end(), QPointF(0.0, 0.0)) * inverse(pts.size());
  }

  // Median the collection of predictions
  void CVLocalizer::med(const std::vector<std::vector<QPointF> >& preds, std::vector<QPointF>& pred) const
  {
    pred.resize(n_points());
    for (uint64_t i = 0; i < n_points(); i ++)
    {
      med(preds[i], pred[i]);
    }
  }
  void CVLocalizer::med(const std::vector<QPointF>& pts, QPointF& pt) const
  {
    std::vector<double> xs(pts.size()), ys(pts.size());
    for (uint64_t i = 0; i < pts.size(); i ++)
    {
      xs[i] = pts[i].x();
      ys[i] = pts[i].y();
    }

    std::sort(xs.begin(), xs.end());
    std::sort(ys.begin(), ys.end());

    pt = QPointF(xs[xs.size() / 2], ys[ys.size() / 2]);
  }

  // Compute the normalized distances [0.0 - 1.0] between
  //	each ground truth keypoints and its predicted points.
  void CVLocalizer::evaluate(  
      const std::vector<std::string>& ifiles, const std::vector<std::string>& gfiles,
      CVDetector& detector,
      std::vector<Histogram>& histos, Histogram& histo) const
  {
    histo = Histogram(100, 0.0, 1.0);
    histos = std::vector<Histogram>(n_points(), histo);                

    // Process each image ...
    for (uint64_t i = 0; i < ifiles.size(); i ++)
    {
      const std::string& ifile = ifiles[i];
      const std::string& gfile = gfiles[i];

      // Load the image and the ground truth
      if (detector.load(ifile, gfile) == false)
      {
        bob::core::warn << "Failed to load image <" << ifile << "> or ground truth <" << gfile << ">!" << std::endl;
        continue;
      }

      // Locate keypoints (on the detections that did not fail)
      Timer timer;				

      std::vector<detection_t> detections;
      detector.scan(detections);

      Object object;
      for (std::vector<detection_t>::const_iterator it = detections.begin(); it != detections.end(); ++ it)
        if (detector.match(*it, object) == true)
        {
          std::vector<QPointF> dt_points;
          if (locate(detector, it->second.first, dt_points) == false)
          {
            bob::core::warn << "Failed to localize the keypoints for the <" << ifile << "> image!" << std::endl;
            continue;
          }

          evaluate(object, dt_points, histos, histo);
        }

      bob::core::info
        << "Image [" << (i + 1) << "/" << ifiles.size() 
        << "]: localized in " << timer.elapsed() << "s." << std::endl;
    }                
  }

  // Compute the normalized distances [0.0 - 1.0] between
  //	each ground truth keypoints and its predicted points.
  void CVLocalizer::evaluate(  
      const std::vector<std::string>& gfiles, const std::vector<std::string>& pfiles,
      std::vector<Histogram>& histos, Histogram& histo) const
  {
    histo = Histogram(100, 0.0, 1.0);
    histos = std::vector<Histogram>(n_points(), histo);                

    // Process each object ...
    for (uint64_t i = 0; i < gfiles.size(); i ++)
    {
      const std::string& gfile = gfiles[i];
      const std::string& pfile = pfiles[i];

      // Load the ground truth and the predictions
      std::vector<Object> gobjects, pobjects;
      if (Object::load(gfile, gobjects) == false)
      {
        bob::core::warn << "Failed to load ground truth <" << gfile << ">!" << std::endl;
        continue;
      }
      if (Object::load(pfile, pobjects) == false)
      {
        bob::core::warn << "Failed to load predictions <" << pfile << ">!" << std::endl;
        continue;
      }

      if (gobjects.size() != pobjects.size())
      {
        bob::core::warn << "Different number of predictions for ground truth <" << gfile << ">!" << std::endl;
        continue;
      }

      // Compare each ground truth with its associated prediction
      Timer timer;				

      for (uint64_t k = 0; k < gobjects.size(); k ++)
      {
        std::vector<QPointF> dt_points(n_points());
        for (uint64_t p = 0; p < n_points(); p ++)
        {
          const std::string name = param().m_labels[p];

          Keypoint keypoint;
          if (pobjects[k].find(name, keypoint) == false)
          {
            bob::core::warn << "Failed to find point <" << name << "> " << "for predictions <" << pfile << ">!" << std::endl;
            continue;
          }

          dt_points[p] = keypoint.m_point;
        }

        evaluate(gobjects[k], dt_points, histos, histo);
      }

      bob::core::info
        << "Ground truth [" << (i + 1) << "/" << gfiles.size() 
        << "]: evaluated in " << timer.elapsed() << "s." << std::endl;
    }                
  }

  // Compute the normalized distances [0.0 - 1.0] between
  //	each ground truth keypoint and its predicted points <dt_points>
  bool CVLocalizer::evaluate(	
      const Object& gt_object, const std::vector<QPointF>& dt_points,
      std::vector<Histogram>& histos, Histogram& histo) const
  {
    // Compute the normalization factor
    //                const double scale = 0.25 * (gt_object.bbx().width() + gt_object.bbx().height());
    //                const double norm = inverse(scale);

    //  --- the distance between the eyes for facial feature
    Keypoint leye, reye;
    gt_object.find("leye", leye);
    gt_object.find("reye", reye);

    const double scale = distance(leye.m_point, reye.m_point);
    const double norm = inverse(scale);

    // Compute and normalize the distance between predictions and ground truths
    double sum_dist = 0.0;
    for (uint64_t ipt = 0; ipt < dt_points.size(); ipt ++)
    {
      Keypoint keypoint;
      if (gt_object.find(param().m_labels[ipt], keypoint) == false)
      {
        return false;
      }

      const double dist = range(norm * distance(keypoint.m_point, dt_points[ipt]), 0.0, 1.0);
      histos[ipt].add(dist);
      sum_dist += dist;
    }

    histo.add(sum_dist * inverse(dt_points.size()));

    return true;
  }	

  // Save the model back to file
  void CVLocalizer::save(const std::string& filename) const
  {
    m_model->save(filename);
  }

}}
