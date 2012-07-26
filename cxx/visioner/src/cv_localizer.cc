#include "visioner/cv/cv_localizer.h"
#include "visioner/model/mdecoder.h"
#include "visioner/util/timer.h"

namespace bob { namespace visioner {

  // Constructor
  CVLocalizer::CVLocalizer()
    :	m_type(MultipleShots_Median)
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
        log_error("CVLocalizer", "decode_var") << po_desc << "\n";
        exit(EXIT_FAILURE);
      }

      var = po_vm[var_name].as<T>();
    }

  void CVLocalizer::add_options(boost::program_options::options_description& po_desc) const
  {
    po_desc.add_options()

      ("localize_model", boost::program_options::value<std::string>(),
       "localization: keypoint localization model")
      ("localize_method", boost::program_options::value<std::string>()->default_value("mshots+med"),
       "localization: method (1shot, mshots+avg, mshots+med");
  }

  bool CVLocalizer::decode(const boost::program_options::options_description& po_desc,
      boost::program_options::variables_map& po_vm)
  {
    // Load the localization model
    const std::string cmd_model = po_vm["localize_model"].as<std::string>();
    if (Model::load(cmd_model, m_model) == false)
    {
      log_error("CVLocalizer", "decode") 
        << "Failed to load the localization model <" << cmd_model << ">!\n";
      return false;
    }
    if (valid_model() == false)
    {
      log_error("CVLocalizer", "decode") << "Invalid model!\n";
      return false;
    }

    // Decode parameters
    string_t cmd_method;
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
      log_error("CVLocalizer", "decode") << "Invalid localization method!\n";
      return false;
    }

    // OK
    return true;
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
  bool CVLocalizer::locate(const CVDetector& detector, const rect_t& reg, points_t& dt_points) const
  {
    // Check the sub-window
    const sw_t sw = detector.ipyramid().map(reg, param());
    if (detector.ipyramid().check(sw, param()) == false)
    {
      return false;
    }

    // Collect predictions (not projected)
    std::vector<points_t> preds(n_points());
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
    points_t pred;                
    switch (m_type)
    {
      case SingleShot:
        for (index_t i = 0; i < n_points(); i ++)
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
      const CVDetector& detector, const sw_t& seed_sw, 
      int n_ds, int n_dx, int dx, int n_dy, int dy, std::vector<points_t>& preds) const
  {       
    std::vector<sws_t> ssws = detector.ipyramid().neighbours(
        seed_sw, n_ds, n_dx, dx, n_dy, dy, param());

    // Process each scale ...
    for (index_t ss = 0; ss < ssws.size(); ss ++)
    {
      const sws_t& sws = ssws[ss];
      if (sws.empty() == true)
      {
        continue;
      }

      const ipscale_t& ip = detector.ipyramid()[sws[0].m_s];                                                
      m_model->preprocess(ip);

      // Process each sub-window at this scale ...
      const scalar_t inv_scale = ip.m_inv_scale;                          
      for (index_t s = 0; s < sws.size(); s ++)
      {
        const sw_t& sw = sws[s];

        for (index_t i = 0; i < n_points(); i ++)
        {
          const scalar_t x = m_model->score(2 * i + 0, sw.m_x, sw.m_y) * param().m_cols;
          const scalar_t y = m_model->score(2 * i + 1, sw.m_x, sw.m_y) * param().m_rows;

          preds[i].push_back(point_t(inv_scale * (sw.m_x + x), 
                inv_scale * (sw.m_y + y)));
        }
      }
    }
  }

  // Average the collection of predictions
  void CVLocalizer::avg(const std::vector<points_t>& preds, points_t& pred) const
  {
    pred.resize(n_points());
    for (index_t i = 0; i < n_points(); i ++)
    {
      avg(preds[i], pred[i]);
    }
  }
  void CVLocalizer::avg(const points_t& pts, point_t& pt) const
  {
    pt = std::accumulate(pts.begin(), pts.end(), point_t(0.0, 0.0)) * inverse(pts.size());
  }

  // Median the collection of predictions
  void CVLocalizer::med(const std::vector<points_t>& preds, points_t& pred) const
  {
    pred.resize(n_points());
    for (index_t i = 0; i < n_points(); i ++)
    {
      med(preds[i], pred[i]);
    }
  }
  void CVLocalizer::med(const points_t& pts, point_t& pt) const
  {
    scalars_t xs(pts.size()), ys(pts.size());
    for (index_t i = 0; i < pts.size(); i ++)
    {
      xs[i] = pts[i].x();
      ys[i] = pts[i].y();
    }

    std::sort(xs.begin(), xs.end());
    std::sort(ys.begin(), ys.end());

    pt = point_t(xs[xs.size() / 2], ys[ys.size() / 2]);
  }

  // Compute the normalized distances [0.0 - 1.0] between
  //	each ground truth keypoints and its predicted points.
  void CVLocalizer::evaluate(  
      const strings_t& ifiles, const strings_t& gfiles,
      CVDetector& detector,
      std::vector<Histogram>& histos, Histogram& histo) const
  {
    histo = Histogram(100, 0.0, 1.0);
    histos = std::vector<Histogram>(n_points(), histo);                

    // Process each image ...
    for (index_t i = 0; i < ifiles.size(); i ++)
    {
      const std::string& ifile = ifiles[i];
      const std::string& gfile = gfiles[i];

      // Load the image and the ground truth
      if (detector.load(ifile, gfile) == false)
      {
        log_warning("CVLocalizer", "evaluate") 
          << "Failed to load image <" << ifile << "> or ground truth <" << gfile << ">!\n";
        continue;
      }

      // Locate keypoints (on the detections that did not fail)
      Timer timer;				

      detections_t detections;
      detector.scan(detections);

      Object object;
      for (detections_const_it it = detections.begin(); it != detections.end(); ++ it)
        if (detector.match(*it, object) == true)
        {
          points_t dt_points;
          if (locate(detector, it->second.first, dt_points) == false)
          {
            log_warning("CVLocalizer", "evaluate")
              << "Failed to localize the keypoints for the <" << ifile << "> image!\n";
            continue;
          }

          evaluate(object, dt_points, histos, histo);
        }

      log_info("CVLocalizer", "evaluate")
        << "Image [" << (i + 1) << "/" << ifiles.size() 
        << "]: localized in " << timer.elapsed() << "s.\n";
    }                
  }

  // Compute the normalized distances [0.0 - 1.0] between
  //	each ground truth keypoints and its predicted points.
  void CVLocalizer::evaluate(  
      const strings_t& gfiles, const strings_t& pfiles,
      std::vector<Histogram>& histos, Histogram& histo) const
  {
    histo = Histogram(100, 0.0, 1.0);
    histos = std::vector<Histogram>(n_points(), histo);                

    // Process each object ...
    for (index_t i = 0; i < gfiles.size(); i ++)
    {
      const std::string& gfile = gfiles[i];
      const std::string& pfile = pfiles[i];

      // Load the ground truth and the predictions
      objects_t gobjects, pobjects;
      if (Object::load(gfile, gobjects) == false)
      {
        log_warning("CVLocalizer", "evaluate") 
          << "Failed to load ground truth <" << gfile << ">!\n";
        continue;
      }
      if (Object::load(pfile, pobjects) == false)
      {
        log_warning("CVLocalizer", "evaluate") 
          << "Failed to load predictions <" << pfile << ">!\n";
        continue;
      }

      if (gobjects.size() != pobjects.size())
      {
        log_warning("CVLocalizer", "evaluate") 
          << "Different number of predictions for ground truth <" << gfile << ">!\n";
        continue;
      }

      // Compare each ground truth with its associated prediction
      Timer timer;				

      for (index_t k = 0; k < gobjects.size(); k ++)
      {
        points_t dt_points(n_points());
        for (index_t p = 0; p < n_points(); p ++)
        {
          const string_t name = param().m_labels[p];

          Keypoint keypoint;
          if (pobjects[k].find(name, keypoint) == false)
          {
            log_warning("CVLocalizer", "evaluate")
              << "Failed to find point <" << name << "> "
              << "for predictions <" << pfile << ">!\n";
            continue;
          }

          dt_points[p] = keypoint.m_point;
        }

        evaluate(gobjects[k], dt_points, histos, histo);
      }

      log_info("CVLocalizer", "evaluate")
        << "Ground truth [" << (i + 1) << "/" << gfiles.size() 
        << "]: evaluated in " << timer.elapsed() << "s.\n";
    }                
  }

  // Compute the normalized distances [0.0 - 1.0] between
  //	each ground truth keypoint and its predicted points <dt_points>
  bool CVLocalizer::evaluate(	
      const Object& gt_object, const points_t& dt_points,
      std::vector<Histogram>& histos, Histogram& histo) const
  {
    // Compute the normalization factor
    //                const scalar_t scale = 0.25 * (gt_object.bbx().width() + gt_object.bbx().height());
    //                const scalar_t norm = inverse(scale);

    //  --- the distance between the eyes for facial feature
    Keypoint leye, reye;
    gt_object.find("leye", leye);
    gt_object.find("reye", reye);

    const scalar_t scale = distance(leye.m_point, reye.m_point);
    const scalar_t norm = inverse(scale);

    // Compute and normalize the distance between predictions and ground truths
    scalar_t sum_dist = 0.0;
    for (index_t ipt = 0; ipt < dt_points.size(); ipt ++)
    {
      Keypoint keypoint;
      if (gt_object.find(param().m_labels[ipt], keypoint) == false)
      {
        return false;
      }

      const scalar_t dist = range(norm * distance(keypoint.m_point, dt_points[ipt]), 0.0, 1.0);
      histos[ipt].add(dist);
      sum_dist += dist;
    }

    histo.add(sum_dist * inverse(dt_points.size()));

    return true;
  }	

}}
