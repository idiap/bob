#include <fstream>

#include "visioner/model/ml.h"

namespace bob { namespace visioner {

  /////////////////////////////////////////////////////////////////////////////////////////
  // Compute the classification/regression error for some
  //      targets and predicted scores
  /////////////////////////////////////////////////////////////////////////////////////////

  scalar_t classification_error(scalar_t target, scalar_t score, scalar_t epsilon)
  {
    const scalar_t edge = target * score;
    return edge > epsilon ? 0.0 : 1.0;
  }

  scalar_t regression_error(scalar_t target, scalar_t score, scalar_t epsilon)
  {
    const scalar_t delta = my_abs(target - score);
    return delta > epsilon ? delta - epsilon : 0.0;
  }

  /////////////////////////////////////////////////////////////////////////////////////////
  // ROC processing
  /////////////////////////////////////////////////////////////////////////////////////////

  // Compute the area under the ROC curve
  scalar_t roc_area(const scalars_t& fars, const scalars_t& tars)
  {
    scalar_t prv_tar = 1.0, prv_far = 1.0, area = 0.0;
    for (index_t i = 0; i < fars.size(); i ++)
    {
      const scalar_t crt_tar = tars[i], crt_far = fars[i];
      area += 0.5 * my_abs(crt_far - prv_far) * (crt_tar + prv_tar);
      prv_tar = crt_tar, prv_far = crt_far;
    }			
    area += 0.5 * my_abs(0.0 - prv_far) * (0.0 + prv_tar);

    return area;
  }

  // Reorder the FARs and TARs such that to make a real curve
  void roc_order(scalars_t& fars, scalars_t& tars)
  {
    std::vector<std::pair<scalar_t, scalar_t> > pts(fars.size());
    for (index_t s = 0; s < fars.size(); s ++)
    {
      pts[s] = std::make_pair(fars[s], tars[s]);
    }

    std::sort(pts.begin(), pts.end(), std::greater<std::pair<scalar_t, scalar_t> >());

    for (index_t s = 0; s < fars.size(); s ++)
    {
      fars[s] = pts[s].first;
      tars[s] = pts[s].second;
    }
  }

  // Trim the ROC curve (remove points that line inside a horizontal segment)
  void roc_trim(scalars_t& fars, scalars_t& tars)
  {
    std::vector<std::pair<scalar_t, scalar_t> > pts(fars.size());
    for (index_t s = 0; s < fars.size(); s ++)
    {
      pts[s] = std::make_pair(fars[s], tars[s]);
    }

    visioner::unique(pts);

    fars.resize(pts.size());
    tars.resize(pts.size());
    for (index_t s = 0; s < fars.size(); s ++)
    {
      fars[s] = pts[s].first;
      tars[s] = pts[s].second;
    }
  }


  // Save the ROC points to file
  bool save_roc(const scalars_t& fars, const scalars_t& tars, const string_t& path)
  {
    std::ofstream out(path.c_str());
    if (out.is_open() == false)
    {
      return false;
    }

    for (index_t i = 0; i < tars.size(); i ++)
    {
      out << tars[i] << "\t" << fars[i] << "\n";
    }		
    out.close();

    return true;
  }

}}
