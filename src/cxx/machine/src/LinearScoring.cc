#include "machine/LinearScoring.h"
#include "math/linear.h"

namespace Torch { namespace machine {

  namespace detail {

    void linearScoring(std::vector<blitz::Array<double,1>*>& models,
                       const blitz::Array<double,1>& ubm_mean,
                       const blitz::Array<double,1>& ubm_variance,
                       std::vector<Torch::machine::GMMStats*>& test_stats,
                       std::vector<blitz::Array<double,1>*>& test_channelOffset,
                       bool frame_length_normalisation,
                       blitz::Array<double,2>& scores) 
    {
      int C = test_stats[0]->sumPx.extent(0);
      int D = test_stats[0]->sumPx.extent(1);
      int CD = C*D;
      int Tt = test_stats.size();
      int Tm = models.size();

      blitz::Array<double,2> A(Tm, CD);
      blitz::Array<double,2> B(CD, Tt);

      // 1) Compute A
      for(int t=0; t<Tm; ++t) {
        blitz::Array<double, 1> tmp = A(t, blitz::Range::all());
        tmp = (*(models[t]) - ubm_mean) / ubm_variance;
      }

      // 2) Compute B
      if(test_channelOffset[0] == NULL) {
        for(int t=0; t<Tt; ++t) 
          for(int s=0; s<CD; ++s)
            B(s, t) = test_stats[t]->sumPx(s/D, s%D) - (ubm_mean(s) * test_stats[t]->n(s/D));
      }
      else {
        Torch::core::array::assertSameDimensionLength(test_channelOffset[0]->extent(0), Tt);
        Torch::core::array::assertSameDimensionLength(test_channelOffset[0]->extent(1), CD);
        
        for(int t=0; t<Tt; ++t) 
          for(int s=0; s<CD; ++s) 
            B(s, t) = test_stats[t]->sumPx(s/D, s%D) - (test_stats[t]->n(s/D) * (ubm_mean(s) + ((*(test_channelOffset[t]))(s))));
      }

      // Apply the normalization if needed
      if(frame_length_normalisation) {
        for(int t=0; t<Tt; ++t) {
          double sum_N = blitz::sum(test_stats[t]->n) * D;
          blitz::Array<double, 1> v_t = B(blitz::Range::all(),t);

          if (sum_N <= std::numeric_limits<double>::epsilon() && sum_N >= -std::numeric_limits<double>::epsilon())
            v_t = 0;
          else 
            v_t /= sum_N;
        }
      }

      // 3) Compute LLR
      scores.resize(Tm, Tt);
      Torch::math::prod(A, B, scores);
    } 
  }


  void linearScoring(std::vector<blitz::Array<double,1>*>& models,
                     const blitz::Array<double,1>& ubm_mean, const blitz::Array<double,1>& ubm_variance,
                     std::vector<Torch::machine::GMMStats*>& test_stats,
                     std::vector<blitz::Array<double,1>*>& test_channelOffset,
                     bool frame_length_normalisation,
                     blitz::Array<double, 2>& scores)
  {
    detail::linearScoring(models, ubm_mean, ubm_variance, test_stats, test_channelOffset, frame_length_normalisation, scores);
  }

  void linearScoring(std::vector<blitz::Array<double,1>*>& models,
                     const blitz::Array<double,1>& ubm_mean, const blitz::Array<double,1>& ubm_variance,
                     std::vector<Torch::machine::GMMStats*>& test_stats,
                     bool frame_length_normalisation,
                     blitz::Array<double, 2>& scores)
  {
    std::vector<blitz::Array<double,1>*> test_channelOffset;
    for(size_t i=0; i<test_stats.size(); ++i)
      test_channelOffset.push_back(NULL);
    detail::linearScoring(models, ubm_mean, ubm_variance, test_stats, test_channelOffset, frame_length_normalisation, scores);
  }

  void linearScoring(std::vector<Torch::machine::GMMMachine*>& models,
                     const Torch::machine::GMMMachine& ubm,
                     std::vector<Torch::machine::GMMStats*>& test_stats,
                     bool frame_length_normalisation,
                     blitz::Array<double, 2>& scores) 
  {
    int C = test_stats[0]->sumPx.extent(0);
    int D = test_stats[0]->sumPx.extent(1);
    int CD = C*D;
    std::vector<blitz::Array<double,1>*> models_b;
    // Allocate and get the mean supervector
    for(size_t i=0; i<models.size(); ++i) {
      blitz::Array<double,1> *mod = new blitz::Array<double,1>(CD);
      models[i]->getMeanSupervector(*mod);
      models_b.push_back(mod);
    }
    const blitz::Array<double,1>& ubm_mean = ubm.getMeanSupervector();
    const blitz::Array<double,1>& ubm_variance = ubm.getVarianceSupervector();
    std::vector<blitz::Array<double,1>*> test_channelOffset;
    for(size_t i=0; i<test_stats.size(); ++i)
      test_channelOffset.push_back(NULL);
    detail::linearScoring(models_b, ubm_mean, ubm_variance, test_stats, test_channelOffset, frame_length_normalisation, scores);
    // Clear the Mean Supervector
    for(size_t i=0; i<models.size(); ++i) 
      delete models_b[i];
  }


  /**
   * Linear scoring: LLR=(model-m_ubm)*S_ubm^-1*(F-N*(m_ubm + channeloffset))^T
   * model = models[i].getMeanSupervector
   * m_ubm = ubm.getMeanSupervector
   * S_ubm = ubm.getVarianceSupervector
   * F     = test_stats[i].sumPx
   * N     = test_stats[i].n
   *
   * The computation is done in 3 steps:
   * 1) Compute A=(model-m_ubm)*S_ubm^-1
   * 2) Compute B=(F-N*(m_ubm + channeloffset))^T
   * 3) Compute LLR = A*B (* is the matrix product)
   */
  void linearScoring(std::vector<Torch::machine::GMMMachine*>& models,
                     Torch::machine::GMMMachine& ubm,
                     std::vector<Torch::machine::GMMStats*>& test_stats,
                     blitz::Array<double, 2>* test_channelOffset,
                     bool frame_length_normalisation,
                     blitz::Array<double, 2>& scores) {
    int C = models[0]->getNGaussians();
    int Dx = models[0]->getNInputs();
    int S = C*Dx;
    int Tm = models.size();
    int Tt = test_stats.size();

    blitz::Array<double, 2> A(Tm, S);
    blitz::Array<double, 2> B(S, Tt);

    
    blitz::Array<double, 1> ubm_meanSupervector;
    ubm.getMeanSupervector(ubm_meanSupervector);
  
    blitz::Array<double, 1> ubm_varianceSupervector;
    ubm.getVarianceSupervector(ubm_varianceSupervector);

    // 1) Compute A
    for(int t = 0; t < Tm; t++) {
      blitz::Array<double, 1> model_meanSupervector;
      models[t]->getMeanSupervector(model_meanSupervector);
      
      blitz::Array<double, 1> tmp = A(t, blitz::Range::all());
      tmp = (model_meanSupervector - ubm_meanSupervector) / ubm_varianceSupervector;
    }


    // 2) Compute B
    if (test_channelOffset == NULL) {
      for(int t = 0; t < Tt; t++) {
        for(int s = 0; s < S; s++) {
          B(s, t) = test_stats[t]->sumPx(s/Dx, s%Dx) - (ubm_meanSupervector(s) * test_stats[t]->n(s/Dx));
        }
      }
    }
    else {
      Torch::core::array::assertSameDimensionLength(test_channelOffset->extent(0), Tt);
      Torch::core::array::assertSameDimensionLength(test_channelOffset->extent(1), S);
      
      for(int t = 0; t < Tt; t++) {
        for(int s = 0; s < S; s++) {
          B(s, t) = test_stats[t]->sumPx(s/Dx, s%Dx) - (test_stats[t]->n(s/Dx) * (ubm_meanSupervector(s) + (*test_channelOffset)(t,s)));
        }
      }
    }

    // Apply the normalization if needed
    if (frame_length_normalisation) {
      for (int t = 0; t < Tt; t++) {
        double sum_N = blitz::sum(test_stats[t]->n) * Dx;
        blitz::Array<double, 1> v_t = B(blitz::Range::all(),t);

        if (sum_N <= std::numeric_limits<double>::epsilon() && sum_N >= -std::numeric_limits<double>::epsilon()) {
          v_t = 0;
        }
        else {
          blitz::firstIndex i;
          v_t = v_t(i) / sum_N;
        }
      }
    }

    // 3) Compute LLR
    scores.resize(Tm, Tt);
    Torch::math::prod(A, B, scores);
  }


}}
