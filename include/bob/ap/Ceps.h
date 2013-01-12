/**
 * @file bob/ap/Ceps.h
 * @date Wed Jan 11:10:20 2013 +0200
 * @author Elie Khoury <Elie.Khoury@idiap.ch>
 *
 * @brief Implement Linear and Mel Frequency Cepstral Coefficients
 * functions (MFCC and LFCC)
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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


#ifndef BOB_AP_CEPS_H
#define BOB_AP_CEPS_H

#include <blitz/array.h>
#include <vector>
#include "bob/sp/FFT1D.h"

const double ENERGY_FLOOR = 1.0;
const double FBANK_OUT_FLOOR = 1.0;

namespace bob {
/**
 * \ingroup libap_api
 * @{
 *
 */
namespace ap {

/**
 * @brief This class is used to test the Ceps class (private methods)
 */
class CepsTest;

/**
 * @brief This class allows the extraction of features from raw audio data.
 * References:
 *  1. SPro tools (http://www.irisa.fr/metiss/guig/spro/spro-4.0.1/spro.html)
 *  2. Wikipedia (http://en.wikipedia.org/wiki/Mel-frequency_cepstrum).
 */
class Ceps
{
  public:
    /**
     * @brief Constructor: Initialize working arrays
     */
    Ceps( double sf, int win_length_ms, int win_shift_ms, int n_filters, int n_ceps,
        double f_min, double f_max, double delta_win);

    /**
     * @brief Get the Cepstral Shape
     */
    blitz::TinyVector<int,2> getCepsShape(const size_t input_length) const;
    blitz::TinyVector<int,2> getCepsShape(const blitz::Array<double,1>& input) const;

    /**
     * @brief Compute Cepstral features
     */
    void CepsAnalysis(const blitz::Array<double,1>& input, blitz::Array<double,2>& output);

    /**
     * @brief Destructor
     */
    virtual ~Ceps();

    /**
     * @brief Getters
     */
    inline double getSampleFrequency() const
    { return m_sf; }
    inline int getWinLengthMs() const
    { return m_win_length_ms; }
    inline int getWinLength() const
    { return m_win_length; }
    inline int getWinShiftMs() const
    { return m_win_shift_ms; }
    inline int getWinShift() const
    { return m_win_shift; }
    inline int getWinSize() const
    { return m_win_size; }
    inline size_t getNFilters() const
    { return m_n_filters; }
    inline size_t getNCeps() const
    { return m_n_ceps; }
    inline double getFMin() const
    { return m_f_min; }
    inline double getFMax() const
    { return m_f_max; }
    inline bool getFbLinear() const
    { return m_fb_linear; }
    inline size_t getDeltaWin() const
    { return m_delta_win; }
    inline double getDctNorm() const
    { return m_dct_norm; }
    inline bool getWithEnergy() const
    { return m_with_energy; }
    inline bool getWithDelta() const
    { return m_with_delta; }
    inline bool getWithDeltaDelta() const
    { return m_with_delta_delta; }
    inline bool getWithDeltaEnergy() const
    { return m_with_delta_energy; }
    inline bool getWithDeltaDeltaEnergy() const
    { return m_with_delta_delta_energy; }

    /**
     * @brief Setters
     */
    void setSampleFrequency(const double sf);
    void setWinLengthMs(int win_length_ms);
    void setWinShiftMs(int win_shift_ms);
    void setNFilters(size_t n_filters);
    void setNCeps(size_t n_ceps);
    inline void setDeltaWin(size_t delta_win)
    { m_delta_win = (int)delta_win; } 
    void setFMin(double f_min);
    void setFMax(double f_max);
    void setFbLinear(bool fb_linear);
    inline void setDctNorm(double dct_norm)
    { m_dct_norm = dct_norm; }
    inline void setWithEnergy(bool with_energy)
    { m_with_energy = with_energy; }
    inline void setWithDelta(bool with_delta)
    { m_with_delta = with_delta; }
    inline void setWithDeltaDelta(bool with_delta_delta)
    { m_with_delta_delta = with_delta_delta; }
    inline void setWithDeltaEnergy(bool with_delta_energy)
    { m_with_delta_energy = with_delta_energy; }
    inline void setWithDeltaDeltaEnergy(bool with_delta_delta_energy)
    { m_with_delta_delta_energy = with_delta_delta_energy; }


  private:
    /**
     * @brief Compute the first derivatives
     */
    void addDelta(blitz::Array<double,2>& frames, int win_size, int n_frames, int frame_size);
    /**
     * @brief Compute the second derivatives
     */
    void addDeltaDelta(blitz::Array<double,2>& frames, int win_size, int n_frames, int frame_size);
    /**
     * @brief Mean Normalisation of the features
     */
    blitz::Array<double,2> dataZeroMean(blitz::Array<double,2>& frames, bool norm_energy, int n_frames, int frame_size);

    static double mel(double f);
    static double melInv(double f);
    void emphasis(blitz::Array<double,1> &data, double a);
    void hammingWindow(blitz::Array<double,1> &data);
    void logFilterBank(blitz::Array<double,1>& x);
    void logTriangularFBank(blitz::Array<double,1>& data);
    double logEnergy(blitz::Array<double,1> &data);
    void transformDCT();
    void initWinSize();
    void initWinLength();
    void initWinShift();
    void initCacheHammingKernel();
    void initCacheDctKernel();
    void initCacheFilterBank();
    void initCachePIndex();
    void initCacheFilters();

    double m_sf;
    int m_win_length_ms;
    int m_win_length;
    int m_win_shift_ms;
    int m_win_shift;
    int m_win_size;
    size_t m_n_filters;
    size_t m_n_ceps;
    double m_f_min;
    double m_f_max;
    int m_delta_win;
    bool m_fb_linear;
    double m_dct_norm;
    bool m_with_energy;
    bool m_with_delta;
    bool m_with_delta_delta;
    bool m_with_delta_energy;
    bool m_with_delta_delta_energy;
    blitz::Array<double,2> m_dct_kernel;
    blitz::Array<double,1> m_hamming_kernel;
    blitz::Array<double,1> m_frame;
    blitz::Array<double,1> m_filters;
    blitz::Array<double,1> m_ceps_coeff;
    blitz::Array<int,1>  m_p_index;
    std::vector<blitz::Array<double,1> > m_filter_bank;
    bob::sp::FFT1D m_fft;
    blitz::Array<std::complex<double>,1>  m_cache_complex1;
    blitz::Array<std::complex<double>,1>  m_cache_complex2;

    friend class TestCeps;
};

class TestCeps
{
  public:
    TestCeps(Ceps& ceps);
    Ceps& m_ceps;

    // Methods to test
    double mel(double f) { return m_ceps.mel(f); }
    double melInv(double f) { return m_ceps.melInv(f); }
    blitz::TinyVector<int,2> getCepsShape(const size_t input_length) const
    { return m_ceps.getCepsShape(input_length); }
    blitz::TinyVector<int,2> getCepsShape(const blitz::Array<double,1>& input) const
    { return m_ceps.getCepsShape(input); }
    blitz::Array<double,1> getFilter(void) {return m_ceps.m_filters;}
    blitz::Array<double,1> getFeatures(void) {return m_ceps.m_ceps_coeff;}

    void CepsAnalysis(const blitz::Array<double,1>& input, blitz::Array<double,2>& ceps_2D)
    { m_ceps.CepsAnalysis(input, ceps_2D);}
    void hammingWindow(blitz::Array<double,1>& data){ m_ceps.hammingWindow(data); }
    void emphasis(blitz::Array<double,1>& data, double a){ m_ceps.emphasis(data, a); }
    void logFilterBank(blitz::Array<double,1>& x){m_ceps.logFilterBank(x);}
    void logTriangularFBank(blitz::Array<double,1>& data){m_ceps.logTriangularFBank(data);}
    double logEnergy(blitz::Array<double,1> &data){return m_ceps.logEnergy(data);}
    void transformDCT(){m_ceps.transformDCT();}
    void addDelta(blitz::Array<double,2>& frames, int win_size, int n_frames, int frame_size){
      m_ceps.addDelta(frames, win_size, n_frames, frame_size);
    }
    void addDeltaDelta(blitz::Array<double,2>& frames, int win_size, int n_frames, int frame_size){
      m_ceps.addDeltaDelta(frames, win_size, n_frames, frame_size);
    }
    blitz::Array<double,2> dataZeroMean(blitz::Array<double,2>& frames, bool norm_energy, int n_frames, int frame_size){
      return m_ceps.dataZeroMean(frames,norm_energy, n_frames, frame_size);
    }
};

}
}

#endif /* BOB_AP_CEPS_H */
