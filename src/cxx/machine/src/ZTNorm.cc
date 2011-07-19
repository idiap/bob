#include <machine/ZTNorm.h>

namespace Torch { namespace machine {
  namespace detail {
    void ztNorm(blitz::Array<double, 2>& eval_tests_on_eval_models,
                blitz::Array<double, 2>& znorm_tests_on_eval_models,
                blitz::Array<double, 2>& eval_tests_on_tnorm_models,
                blitz::Array<double, 2>& znorm_tests_on_tnorm_models,
                blitz::Array<bool, 2>*  znorm_tests_tnorm_models_same_spk_ids,
                blitz::Array<double, 2>& scores) {
      // Rename variables
      blitz::Array<double, 2> A = eval_tests_on_eval_models;
      blitz::Array<double, 2> B = znorm_tests_on_eval_models;
      blitz::Array<double, 2> C = eval_tests_on_tnorm_models;
      blitz::Array<double, 2> D = znorm_tests_on_tnorm_models;

      // Compute the sizes
      int size_enroll = A.extent(1);
      int size_eval = A.extent(0);
      int size_tnorm = C.extent(0);
      int size_znorm = B.extent(1);

      // Check the inputs
      Torch::core::array::assertSameDimensionLength(A.extent(0), size_eval);
      Torch::core::array::assertSameDimensionLength(A.extent(1), size_enroll);

      Torch::core::array::assertSameDimensionLength(B.extent(0), size_eval);
      Torch::core::array::assertSameDimensionLength(B.extent(1), size_znorm);

      Torch::core::array::assertSameDimensionLength(C.extent(0), size_tnorm);
      Torch::core::array::assertSameDimensionLength(C.extent(1), size_enroll);

      Torch::core::array::assertSameDimensionLength(D.extent(0), size_tnorm);
      Torch::core::array::assertSameDimensionLength(D.extent(1), size_znorm);

      if (znorm_tests_tnorm_models_same_spk_ids) {
        Torch::core::array::assertSameDimensionLength(znorm_tests_tnorm_models_same_spk_ids->extent(0), size_tnorm);
        Torch::core::array::assertSameDimensionLength(znorm_tests_tnorm_models_same_spk_ids->extent(1), size_znorm);
      }

      // Declare needed IndexPlaceholder
      blitz::firstIndex i;
      blitz::secondIndex j;

      
      blitz::Array<double, 2> zA(size_eval, size_enroll);
      blitz::Array<double, 1> mean_B(size_eval);

      // Znorm  -->      zA  = (A - mean(B) ) / std(B)    [znorm on oringinal scores]
      mean_B = blitz::mean(B, j);
      zA = (A(i, j) - mean_B(i)) / sqrt(blitz::sum(pow(B(i, j) - mean_B(i), 2) , j) / (size_znorm - 1));

      blitz::Array<double, 1> mean_Dimp(size_tnorm);
      blitz::Array<double, 1> std_Dimp(size_tnorm);

      // Compute mean_Dimp and std_Dimp = D only with impostors
      for(int i = 0; i < size_tnorm; i++) {
        double sum = 0;
        double sumsq = 0;
        double count = 0;
        for(int j = 0; j < size_znorm; j++) {
          bool keep;
          // The second part is never executed if znorm_tests_tnorm_models_same_spk_ids==NULL
          keep = (znorm_tests_tnorm_models_same_spk_ids == NULL) || !(*znorm_tests_tnorm_models_same_spk_ids)(i, j); //tnorm_models_spk_ids(i) != znorm_tests_spk_ids(j);
          
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
      blitz::Array<double, 2> zC(size_tnorm, size_enroll);
      zC = (C(i, j) - mean_Dimp(i)) / std_Dimp(i);

      blitz::Array<double, 1> mean_zC(size_enroll);
      blitz::Array<double, 1> std_zC(size_enroll);
      scores.resize(size_eval, size_enroll);
      
      // ztA = (zA - mean(zC)) / std(zC)  [ztnorm on eval scores]
      mean_zC = blitz::mean(zC(j, i), j);
      std_zC = sqrt(blitz::sum(pow(zC(j, i) - mean_zC(i), 2) , j) / (size_tnorm - 1));
      scores = (zA(i, j) - mean_zC(j)) /  std_zC(j);
    }
  }

  
  void ztNorm(blitz::Array<double, 2>& eval_tests_on_eval_models,
              blitz::Array<double, 2>& znorm_tests_on_eval_models,
              blitz::Array<double, 2>& eval_tests_on_tnorm_models,
              blitz::Array<double, 2>& znorm_tests_on_tnorm_models,
              blitz::Array<bool,   2>& znorm_tests_tnorm_models_same_spk_ids,
              blitz::Array<double, 2>& scores) {
    detail::ztNorm(eval_tests_on_eval_models, znorm_tests_on_eval_models, eval_tests_on_tnorm_models,
                   znorm_tests_on_tnorm_models, &znorm_tests_tnorm_models_same_spk_ids, scores);
  }
  
  void ztNorm(blitz::Array<double, 2>& eval_tests_on_eval_models,
              blitz::Array<double, 2>& znorm_tests_on_eval_models,
              blitz::Array<double, 2>& eval_tests_on_tnorm_models,
              blitz::Array<double, 2>& znorm_tests_on_tnorm_models,
              blitz::Array<double, 2>& scores) {
    detail::ztNorm(eval_tests_on_eval_models, znorm_tests_on_eval_models, eval_tests_on_tnorm_models,
                   znorm_tests_on_tnorm_models, NULL, scores);
  }
}
}