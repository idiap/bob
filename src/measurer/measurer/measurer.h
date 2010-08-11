#ifndef MEASURER_INC
#define MEASURER_INC

#include "core/general.h"

namespace Torch {

struct LabelledMeasure {
  short label;
  double measure;
};

/// this function returns 1 if p1->measure > p2->measure
extern "C" int cmp_labelledmeasure(const void *p1, const void *p2);

double computeTH(LabelledMeasure* measures, int n_size, double dr);

/// Search for the EER and returns the threshold, the FRR and FAR
double computeEER(LabelledMeasure* measures, int n, double* frr, double* far, int number_of_positives_ = -1, bool sort_ = true);

/// Search for the min HTER and returns the threshold, the FRR and FAR
double computeHTER(LabelledMeasure* measures, int n, double* frr, double* far, int number_of_positives_ = -1, bool sort = true, float ratio_far = 1.);

//void computeFaFr(real thrd, Int_real* to_sort, int n, real* frr, real* far, int number_of_clients_ = -1,bool sort = true);

}
#endif
