#include "visioner/cv/cv_classifier.h"
#include "visioner/model/mdecoder.h"
#include "visioner/util/timer.h"
#include "visioner/model/taggers/tagger_object.h"

namespace bob { namespace visioner {

  // Constructor
  CVClassifier::CVClassifier()
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
        log_error("CVClassifier", "decode_var") << po_desc << "\n";
        exit(EXIT_FAILURE);
      }

      var = po_vm[var_name].as<T>();
    }

  void CVClassifier::add_options(boost::program_options::options_description& po_desc) const
  {
    po_desc.add_options()

      ("classify_model", boost::program_options::value<std::string>(),
       "classification: model");
  }

  bool CVClassifier::decode(const boost::program_options::options_description& po_desc,
      boost::program_options::variables_map& po_vm)
  {
    // Load the localization model
    const std::string cmd_model = po_vm["classify_model"].as<std::string>();
    if (Model::load(cmd_model, m_model) == false)
    {
      log_error("CVClassifier", "decode") 
        << "Failed to load the classification model <" << cmd_model << ">!\n";
      return false;
    }
    if (valid_model() == false)
    {
      log_error("CVClassifier", "decode") << "Invalid model!\n";
      return false;
    }

    // Decode parameters
    //

    // OK
    return true;
  }

  // Check the validity of different components
  bool CVClassifier::valid() const
  {
    return valid_model();
  }
  bool CVClassifier::valid_model() const
  {
    return	m_model.get() != 0 &&
      m_model->n_outputs() == param().m_labels.size();
  }

  // Predict the object label of the keypoints in the <reg> region.
  bool CVClassifier::classify(const CVDetector& detector, const rect_t& reg, index_t& dt_label) const
  {
    // Check the sub-window
    const sw_t sw = detector.ipyramid().map(reg, param());
    if (detector.ipyramid().check(sw, param()) == false)
    {
      return false;
    }

    // Evaluate the sub-window
    const ipscale_t& ip = detector.ipyramid()[sw.m_s];
    m_model->preprocess(ip);

    scalars_t scores(n_classes());                
    for (index_t o = 0; o < n_classes(); o ++)
    { 
      scores[o] = m_model->score(o, sw.m_x, sw.m_y);
    }

    dt_label = std::max_element(scores.begin(), scores.end()) - scores.begin();                

    // OK
    return true;
  }

  // Retrieve the ground truth label for the given object
  bool CVClassifier::classify(  const Object& object, index_t& gt_label) const
  {
    const rtagger_t tagger = make_tagger(param());
    const ObjectTagger* obj_tagger = dynamic_cast<const ObjectTagger*>(tagger.get());
    if (obj_tagger == 0)
    {
      log_warning("CVClassifier", "classify") 
        << "Invalid classification tagger!\n";
      return false;
    }

    const int ilabel = obj_tagger->find(object);
    gt_label = ilabel;
    return ilabel >= 0;
  }

  // Compute the confusion matrix considering the 
  //      ground truth and the predicted labels.
  void CVClassifier::evaluate(
      const strings_t& ifiles, const strings_t& gfiles,
      CVDetector& detector, 
      index_mat_t& hits_mat, indices_t& hits_cnt) const
  {
    hits_mat.resize(n_classes(), n_classes());
    hits_mat.fill(0);

    hits_cnt.resize(n_classes());
    std::fill(hits_cnt.begin(), hits_cnt.end(), 0);

    const rtagger_t tagger = make_tagger(param());
    const ObjectTagger* obj_tagger = dynamic_cast<const ObjectTagger*>(tagger.get());
    if (obj_tagger == 0)
    {
      log_warning("CVClassifier", "evaluate") 
        << "Invalid classification tagger!\n";
      return;
    }

    // Process each image ...        
    for (index_t i = 0; i < ifiles.size(); i ++)
    {
      const std::string& ifile = ifiles[i];
      const std::string& gfile = gfiles[i];

      // Load the image and the ground truth
      if (detector.load(ifile, gfile) == false)
      {
        log_warning("CVClassifier", "evaluate") 
          << "Failed to load image <" << ifile << "> or ground truth <" << gfile << ">!\n";
        continue;
      }

      // Classify sub-windows (on the detections that did not fail)
      Timer timer;				

      detections_t detections;
      detector.scan(detections);

      Object object;
      for (detections_const_it it = detections.begin(); it != detections.end(); ++ it)
        if (detector.match(*it, object) == true)
        {
          const int gt_label = obj_tagger->find(object);
          if (gt_label < 0)
          {
            continue;
          }

          index_t dt_label = 0;
          if (classify(detector, object.bbx(), dt_label) == false)
          {
            log_warning("CVClassifier", "evaluate")
              << "Failed to classify a sub-window for the <" << ifile << "> image!\n";
            continue;
          }

          hits_mat(gt_label, dt_label) ++;
          hits_cnt[gt_label] ++;
        }

      // Debug
      log_info("CVClassifier", "evaluate")
        << "Image [" << (i + 1) << "/" << ifiles.size() 
        << "]: classified in " << timer.elapsed() << "s.\n";
    }
  }

}}
