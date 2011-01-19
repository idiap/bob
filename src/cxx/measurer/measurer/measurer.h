#ifndef TORCH5SPRO_MEASURER_H_
#define TORCH5SPRO_MEASURER_H_

#include "core/general.h"

namespace Torch {

  /**
   * \defgroup libmeasurer_api libMeasurer API
   * @{
   *
   *  The libMeasurer API.
   */

  /**
   * A structure to represent a measure with a particular label
   */
  struct LabelledMeasure {
    short label;
    double measure;
  };

  /**
   * @param p1 the first labelled measure
   * @param p2 the second labelled measure
   * @return Returns 1 if p1->measure > p2->measure
   */
  extern "C" int cmp_labelledmeasure(const void *p1, const void *p2);

  double computeTH(LabelledMeasure* measures, int n_size, double dr);

  /**
   * @brief Search for the EER and returns the threshold, the FRR and FAR
   * @param measures the array of measures to consider
   * @param n the number of measures
   * @param frr a pointer to the FRR (updated by this function)
   * @param far a pointer to the FAR (updated by this function)
   * @param number_of_positives_ the number of positive samples
   * @param sort_ indicates if a sort should be performed
   * @return Returns the EER
   */
  double computeEER(LabelledMeasure* measures, int n, double* frr, 
      double* far, int number_of_positives_ = -1, bool sort_ = true);

  /**
   * @brief Search for the min HTER and returns the threshold, 
   * the FRR and FAR
   * @param measures the array of measures to consider
   * @param n the number of measures
   * @param frr a pointer to the FRR (updated by this function)
   * @param far a pointer to the FAR (updated by this function)
   * @param number_of_positives_ the number of positive samples
   * @param sort_ indicates if a sort should be performed
   * @param ratio_far the FAR ratio for the cost computation
   * @return Returns the HTER
   */
  double computeHTER(LabelledMeasure* measures, int n, double* frr, 
      double* far, int number_of_positives_ = -1, bool sort = true, 
      float ratio_far = 1.);

  //void computeFaFr(real thrd, Int_real* to_sort, int n, real* frr, 
  //    real* far, int number_of_clients_ = -1,bool sort = true);

  /**
   * @}
   */

}

#endif /* TORCH5SPRO_MEASURER_H_ */
