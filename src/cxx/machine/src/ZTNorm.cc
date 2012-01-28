/**
 * @file src/cxx/machine/src/ZTNorm.cc
 * @date Tue Jul 19 15:33:20 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
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

#include "machine/ZTNorm.h"
#include "core/array_assert.h"

namespace bob { 
namespace machine {

namespace detail {
  void ztNorm(const blitz::Array<double, 2>& rawscores_probes_vs_models,
              const blitz::Array<double, 2>& rawscores_zprobes_vs_models,
              const blitz::Array<double, 2>& rawscores_probes_vs_tmodels,
              const blitz::Array<double, 2>& rawscores_zprobes_vs_tmodels,
              const blitz::Array<bool, 2>*  mask_zprobes_vs_tmodels_istruetrial,
              blitz::Array<double, 2>& scores) 
  {
    // Rename variables
    const blitz::Array<double, 2>& A = rawscores_probes_vs_models;
    const blitz::Array<double, 2>& B = rawscores_zprobes_vs_models;
    const blitz::Array<double, 2>& C = rawscores_probes_vs_tmodels;
    const blitz::Array<double, 2>& D = rawscores_zprobes_vs_tmodels;

    // Compute the sizes
    int size_eval  = A.extent(0);
    int size_enrol = A.extent(1);
    int size_tnorm = C.extent(0);
    int size_znorm = B.extent(1);

    // Check the inputs
    bob::core::array::assertSameDimensionLength(A.extent(0), size_eval);
    bob::core::array::assertSameDimensionLength(A.extent(1), size_enrol);

    bob::core::array::assertSameDimensionLength(B.extent(0), size_eval);
    bob::core::array::assertSameDimensionLength(B.extent(1), size_znorm);

    bob::core::array::assertSameDimensionLength(C.extent(0), size_tnorm);
    bob::core::array::assertSameDimensionLength(C.extent(1), size_enrol);

    bob::core::array::assertSameDimensionLength(D.extent(0), size_tnorm);
    bob::core::array::assertSameDimensionLength(D.extent(1), size_znorm);

    if (mask_zprobes_vs_tmodels_istruetrial) {
      bob::core::array::assertSameDimensionLength(mask_zprobes_vs_tmodels_istruetrial->extent(0), size_tnorm);
      bob::core::array::assertSameDimensionLength(mask_zprobes_vs_tmodels_istruetrial->extent(1), size_znorm);
    }

    bob::core::array::assertSameDimensionLength(scores.extent(0), size_eval);
    bob::core::array::assertSameDimensionLength(scores.extent(1), size_enrol);

    // Declare needed IndexPlaceholder
    blitz::firstIndex i;
    blitz::secondIndex j;

    
    // Znorm  -->      zA  = (A - mean(B) ) / std(B)    [znorm on oringinal scores]
    // mean(B)
    blitz::Array<double, 1> mean_B(blitz::mean(B, j));
    // std(B)
    blitz::Array<double, 2> B2n(B.shape());
    B2n = blitz::pow2(B(i, j) - mean_B(i));
    blitz::Array<double, 1> std_B(B.extent(0));
    std_B = blitz::sqrt(blitz::sum(B2n, j) / (size_znorm - 1));
    // zA
    blitz::Array<double, 2> zA(A.shape());
    zA = (A(i, j) - mean_B(i)) / std_B(i);

    blitz::Array<double, 1> mean_Dimp(size_tnorm);
    blitz::Array<double, 1> std_Dimp(size_tnorm);

    // Compute mean_Dimp and std_Dimp = D only with impostors
    for(int i = 0; i < size_tnorm; ++i) {
      double sum = 0;
      double sumsq = 0;
      double count = 0;
      for(int j = 0; j < size_znorm; ++j) {
        bool keep;
        // The second part is never executed if mask_zprobes_vs_tmodels_istruetrial==NULL
        keep = (mask_zprobes_vs_tmodels_istruetrial == NULL) || !(*mask_zprobes_vs_tmodels_istruetrial)(i, j); //tnorm_models_spk_ids(i) != znorm_tests_spk_ids(j);
        
        double value = keep * D(i, j);
        sum += value;
        sumsq += value*value;
        count += keep;
      }

      // TODO We assume that count is > 0
      double mean = sum / count;
      mean_Dimp(i) = mean;
      std_Dimp(i) = sqrt((sumsq - count * mean * mean) / (count -1));
    }

    // zC  = (C - mean(D)) / std(D)     [znorm the tnorm scores]
    blitz::Array<double, 2> zC(size_tnorm, size_enrol);
    zC = (C(i, j) - mean_Dimp(i)) / std_Dimp(i);

    blitz::Array<double, 1> mean_zC(size_enrol);
    blitz::Array<double, 1> std_zC(size_enrol);
    
    // ztA = (zA - mean(zC)) / std(zC)  [ztnorm on eval scores]
    mean_zC = blitz::mean(zC(j, i), j);
    std_zC = sqrt(blitz::sum(pow(zC(j, i) - mean_zC(i), 2) , j) / (size_tnorm - 1));
    scores = (zA(i, j) - mean_zC(j)) /  std_zC(j);
  }
}


void ztNorm(const blitz::Array<double, 2>& rawscores_probes_vs_models,
            const blitz::Array<double, 2>& rawscores_zprobes_vs_models,
            const blitz::Array<double, 2>& rawscores_probes_vs_tmodels,
            const blitz::Array<double, 2>& rawscores_zprobes_vs_tmodels,
            const blitz::Array<bool,   2>& mask_zprobes_vs_tmodels_istruetrial,
            blitz::Array<double, 2>& scores) 
{
  detail::ztNorm(rawscores_probes_vs_models, rawscores_zprobes_vs_models, rawscores_probes_vs_tmodels,
                 rawscores_zprobes_vs_tmodels, &mask_zprobes_vs_tmodels_istruetrial, scores);
}

void ztNorm(const blitz::Array<double, 2>& rawscores_probes_vs_models,
            const blitz::Array<double, 2>& rawscores_zprobes_vs_models,
            const blitz::Array<double, 2>& rawscores_probes_vs_tmodels,
            const blitz::Array<double, 2>& rawscores_zprobes_vs_tmodels,
            blitz::Array<double, 2>& scores) 
{
  detail::ztNorm(rawscores_probes_vs_models, rawscores_zprobes_vs_models, rawscores_probes_vs_tmodels,
                 rawscores_zprobes_vs_tmodels, NULL, scores);
}


}}
