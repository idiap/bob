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
     * @brief Constructor. Initializes working arrays
     */
    Ceps(double sampling_frequency, double win_length_ms=20., double win_shift_ms=10.,
      size_t n_filters=24, size_t n_ceps=19, double f_min=0., 
      double f_max=4000., size_t delta_win=2, double pre_emphasis_coef=0.95,
      bool mel_scale=true, bool dct_norm=false);

    /**
     * @brief Gets the Cepstral features shape for a given input/input length
     */
    blitz::TinyVector<int,2> getCepsShape(const size_t input_length) const;
    blitz::TinyVector<int,2> getCepsShape(const blitz::Array<double,1>& input) const;

    /**
     * @brief Computes Cepstral features
     */
    void operator()(const blitz::Array<double,1>& input, blitz::Array<double,2>& output);

    /**
     * @brief Destructor
     */
    virtual ~Ceps();

    /**
     * @brief Returns the sampling frequency/frequency rate
     */
    inline double getSamplingFrequency() const
    { return m_sampling_frequency; }
    /**
     * @brief Returns the window length in miliseconds
     */
    inline double getWinLengthMs() const
    { return m_win_length_ms; }
    /**
     * @brief Returns the window length in number of samples
     */
    inline size_t getWinLength() const
    { return m_win_length; }
    /**
     * @brief Returns the window shift in miliseconds
     */
    inline double getWinShiftMs() const
    { return m_win_shift_ms; }
    /**
     * @brief Returns the window shift in number of samples
     */
    inline size_t getWinShift() const
    { return m_win_shift; }
    /**
     * @brief Returns the number of filters used in the filter bank.
     */
    inline size_t getNFilters() const
    { return m_n_filters; }
    /**
     * @brief Returns the number of cepstral coefficient to keep
     */
    inline size_t getNCeps() const
    { return m_n_ceps; }
    /**
     * @brief Returns the frequency of the lowest triangular filter in the
     * filter bank
     */
    inline double getFMin() const
    { return m_f_min; }
    /**
     * @brief Returns the frequency of the highest triangular filter in the
     * filter bank
     */
    inline double getFMax() const
    { return m_f_max; }
    /**
     * @brief Tells whether the frequencies of the filters in the filter bank
     * are taken from the linear or the Mel scale
     */
    inline bool getMelScale() const
    { return m_mel_scale; }
    /**
     * @brief Rerturns the size of the window used to compute first and second
     * order derivatives
     */
    inline size_t getDeltaWin() const
    { return m_delta_win; }
    /**
     * @brief Returns the pre-emphasis coefficient.
     */
    inline double getPreEmphasisCoeff() const
    { return m_pre_emphasis_coeff; }
    /**
     * @brief Tells whether the DCT coefficients are normalized or not
     */
    inline bool getDctNorm() const
    { return m_dct_norm; }
    /**
     * @brief Tells whether the energy is added to the cepstral coefficients 
     * or not
     */
    inline bool getWithEnergy() const
    { return m_with_energy; }
    /**
     * @brief Tells whether the first order derivatives are added to the 
     * cepstral coefficients or not
     */
    inline bool getWithDelta() const
    { return m_with_delta; }
    /**
     * @brief Tells whether the second order derivatives are added to the 
     * cepstral coefficients or not
     */
    inline bool getWithDeltaDelta() const
    { return m_with_delta_delta; }

    /**
     * @brief Sets the sampling frequency/frequency rate
     */
    void setSamplingFrequency(const double sampling_frequency);
    /**
     * @brief Sets the window length in miliseconds
     */
    void setWinLengthMs(double win_length_ms);
    /**
     * @brief Sets the window shift in miliseconds
     */
    void setWinShiftMs(double win_shift_ms);
    /**
     * @brief Sets the number of filters used in the filter bank.
     */
    void setNFilters(size_t n_filters);
    /**
     * @brief Returns the number of cepstral coefficient to keep
     */
    void setNCeps(size_t n_ceps);
    /**
     * @brief Sets the size of the window used to compute first and second
     * order derivatives
     */
    inline void setDeltaWin(size_t delta_win)
    { m_delta_win = delta_win; } 
    /**
     * @brief Sets the pre-emphasis coefficient. It should be a value in the 
     * range [0,1].
     */
    inline void setPreEmphasisCoeff(double pre_emphasis_coeff)
    { // TODO: check paramater value is in range [0,1]
      m_pre_emphasis_coeff = pre_emphasis_coeff; }
    /**
     * @brief Returns the frequency of the lowest triangular filter in the
     * filter bank
     */
    void setFMin(double f_min);
    /**
     * @brief Returns the frequency of the highest triangular filter in the
     * filter bank
     */
    void setFMax(double f_max);
    /**
     * @brief Sets whether the frequencies of the filters in the filter bank
     * are taken from the linear or the Mel scale
     */
    void setMelScale(bool mel_scale);
    /**
     * @brief Sets whether the DCT coefficients are normalized or not
     */
    void setDctNorm(bool dct_norm);
    /**
     * @brief Sets whether the energy is added to the cepstral coefficients 
     * or not
     */
    inline void setWithEnergy(bool with_energy)
    { m_with_energy = with_energy; }
    /**
     * @brief Sets whether the first order derivatives are added to the 
     * cepstral coefficients or not
     */
    inline void setWithDelta(bool with_delta)
    { if(!with_delta) m_with_delta_delta = false;
      m_with_delta = with_delta; }
    /**
     * @brief Sets whether the first order derivatives are added to the 
     * cepstral coefficients or not. If enabled, first order derivatives are
     * automatically enabled as well.
     */
    inline void setWithDeltaDelta(bool with_delta_delta)
    { if(with_delta_delta) m_with_delta = true;
      m_with_delta_delta = with_delta_delta; }

  private:
    /**
     * @brief Computes the first order derivative from the given input. 
     * This methods is used to compute both the delta's and double delta's.
     */
    void addDerivative(const blitz::Array<double,2>& input, blitz::Array<double,2>& output) const;
    /**
     * @brief Converts a frequency in Herz to the corresponding one in Mel
     */
    static double herzToMel(double f);
    /**
     * @brief Converts a frequency in Mel to the corresponding one in Herz
     */
    static double melToHerz(double f);
    /**
     * @brief Pre-emphasises the signal by applying the first order equation
     * \f$data_{n} := data_{n} − a*data_{n−1}\f$
     */
    void pre_emphasis(blitz::Array<double,1> &data) const;
    /**
     * @brief Applies the Hamming window to the signal
     */
    void hammingWindow(blitz::Array<double,1> &data) const;

    /**
     * @brief Computes the power-spectrum of the FFT of the input frame and
     * applies the triangular filter bank
     */
    void logFilterBank(blitz::Array<double,1>& x);
    /**
     * @brief Applies the triangular filter bank to the input array and 
     * returns the logarithm of the energy in each band.
     */
    void logTriangularFilterBank(blitz::Array<double,1>& data) const;
    /**
     * @brief Computes the logarithm of the energy
     */
    double logEnergy(blitz::Array<double,1> &data) const;
    /**
     * @brief Applies the DCT to the cepstral features:
     * \f$out[i]=sqrt(2/N)*sum_{j=1}^{N} (in[j]cos(M_PI*i*(j-0.5)/N)\f$
     */
    void applyDct(blitz::Array<double,1>& ceps_row) const;

    void initWinSize();
    void initWinLength();
    void initWinShift();

    void initCacheHammingKernel();
    void initCacheDctKernel();
    void initCacheFilterBank();
    /**
     * @brief Initialize the table m_p_index, which contains the indices of
     * the cut-off frequencies of the triangular filters.. It looks like:
     *
     *                      filter 2
     *                   <------------->
     *                filter 1           filter 4
     *             <----------->       <------------->
     *        | | | | | | | | | | | | | | | | | | | | | ..........
     *         0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9  ..........
     *             ^     ^     ^       ^             ^
     *             |     |     |       |             |
     *            t[0]   |    t[2]     |           t[4]
     *                  t[1]          t[3]
     *
     */
    void initCachePIndex();
    void initCacheFilters();

    double m_sampling_frequency; ///< The sampling frequency
    double m_win_length_ms; ///< The window length in miliseconds 
    size_t m_win_length;
    double m_win_shift_ms;
    size_t m_win_shift;
    size_t m_win_size;
    size_t m_n_filters;
    size_t m_n_ceps;
    double m_f_min;
    double m_f_max;
    size_t m_delta_win;
    double m_pre_emphasis_coeff;
    bool m_mel_scale;
    bool m_dct_norm;
    bool m_with_energy;
    bool m_with_delta;
    bool m_with_delta_delta;
    double m_energy_floor;
    double m_fb_out_floor;
    double m_log_energy_floor;
    double m_log_fb_out_floor;

    blitz::Array<double,2> m_dct_kernel;
    blitz::Array<double,1> m_hamming_kernel;
    blitz::Array<int,1>  m_p_index;
    std::vector<blitz::Array<double,1> > m_filter_bank;
    bob::sp::FFT1D m_fft;

    mutable blitz::Array<double,1> m_cache_frame_d;
    mutable blitz::Array<std::complex<double>,1>  m_cache_frame_c1;
    mutable blitz::Array<std::complex<double>,1>  m_cache_frame_c2;
    mutable blitz::Array<double,1> m_cache_filters;

    friend class TestCeps;
};

class TestCeps
{
  public:
    TestCeps(Ceps& ceps);
    Ceps& m_ceps;

    // Methods to test
    double herzToMel(double f) { return m_ceps.herzToMel(f); }
    double melToHerz(double f) { return m_ceps.melToHerz(f); }
    blitz::TinyVector<int,2> getCepsShape(const size_t input_length) const
    { return m_ceps.getCepsShape(input_length); }
    blitz::TinyVector<int,2> getCepsShape(const blitz::Array<double,1>& input) const
    { return m_ceps.getCepsShape(input); }
    blitz::Array<double,1> getFilterOutput() { return m_ceps.m_cache_filters; }

    void operator()(const blitz::Array<double,1>& input, blitz::Array<double,2>& ceps_2D)
    { m_ceps(input, ceps_2D);}
    void hammingWindow(blitz::Array<double,1>& data){ m_ceps.hammingWindow(data); }
    void pre_emphasis(blitz::Array<double,1>& data){ m_ceps.pre_emphasis(data); }
    void logFilterBank(blitz::Array<double,1>& x){ m_ceps.logFilterBank(x); }
    void logTriangularFilterBank(blitz::Array<double,1>& data){ m_ceps.logTriangularFilterBank(data); }
    double logEnergy(blitz::Array<double,1> &data){ return m_ceps.logEnergy(data); }
    void applyDct(blitz::Array<double,1>& ceps_row) { m_ceps.applyDct(ceps_row); }
};

}
}

#endif /* BOB_AP_CEPS_H */
