/**
 * @file bob/ap/Spectrogram.h
 * @date Wed Jan 11:10:20 2013 +0200
 * @author Elie Khoury <Elie.Khoury@idiap.ch>
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement spectrogram
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


#ifndef BOB_AP_SPECTROGRAM_H
#define BOB_AP_SPECTROGRAM_H

#include <blitz/array.h>
#include <vector>
#include "Energy.h"
#include <bob/core/Exception.h>
#include <bob/sp/FFT1D.h>

namespace bob {
/**
 * \ingroup libap_api
 * @{
 *
 */
namespace ap {

/**
 * @brief This class implements an audio spectrogram extractor
 */
class Spectrogram: public Energy
{
  public:
    /**
     * @brief Constructor. Initializes working arrays
     */
    Spectrogram(const double sampling_frequency, 
      const double win_length_ms=20., const double win_shift_ms=10.,
      const size_t n_filters=24, const double f_min=0., 
      const double f_max=4000., const double pre_emphasis_coeff=0.95,
      bool mel_scale=true);

    /**
     * @brief Copy Constructor
     */
    Spectrogram(const Spectrogram& other); 

    /**
     * @brief Assignment operator
     */
    Spectrogram& operator=(const Spectrogram& other);

    /**
     * @brief Equal to
     */
    bool operator==(const Spectrogram& other) const;

    /**
     * @brief Not equal to
     */
    bool operator!=(const Spectrogram& other) const;

    /**
     * @brief Destructor
     */
    virtual ~Spectrogram();

    /**
     * @brief Gets the output shape for a given input/input length
     */
    virtual blitz::TinyVector<int,2> getShape(const size_t input_length) const;
    virtual blitz::TinyVector<int,2> getShape(const blitz::Array<double,1>& input) const;

    /**
     * @brief Computes the spectrogram
     */
    void operator()(const blitz::Array<double,1>& input, blitz::Array<double,2>& output);

    /**
     * @brief Returns the number of filters used in the filter bank.
     */
    size_t getNFilters() const
    { return m_n_filters; }
    /**
     * @brief Returns the frequency of the lowest triangular filter in the
     * filter bank
     */
    double getFMin() const
    { return m_f_min; }
    /**
     * @brief Returns the frequency of the highest triangular filter in the
     * filter bank
     */
    double getFMax() const
    { return m_f_max; }
    /**
     * @brief Tells whether the frequencies of the filters in the filter bank
     * are taken from the linear or the Mel scale
     */
    bool getMelScale() const
    { return m_mel_scale; }
    /**
     * @brief Returns the pre-emphasis coefficient.
     */
    double getPreEmphasisCoeff() const
    { return m_pre_emphasis_coeff; }
    /**
     * @brief Tells whether we used the energy or not
     */
    bool getEnergyFilter() const
    { return m_energy_filter; }

    /** 
     * @brief Sets the sampling frequency/frequency rate
     */
    virtual void setSamplingFrequency(const double sampling_frequency);
    /** 
     * @brief Sets the window length in miliseconds
     */
    virtual void setWinLengthMs(const double win_length_ms);
    /** 
     * @brief Sets the window shift in miliseconds
     */
    virtual void setWinShiftMs(const double win_shift_ms);

    /**
     * @brief Sets the number of filters used in the filter bank.
     */
    virtual void setNFilters(size_t n_filters);
    /**
     * @brief Sets the pre-emphasis coefficient. It should be a value in the 
     * range [0,1].
     */
    virtual void setPreEmphasisCoeff(double pre_emphasis_coeff)
    {
      if (pre_emphasis_coeff < 0. || pre_emphasis_coeff > 1.)
        throw bob::core::InvalidArgumentException("pre_emphasis_coeff",
          pre_emphasis_coeff, 0., 1.);
      m_pre_emphasis_coeff = pre_emphasis_coeff; }
    /**
     * @brief Returns the frequency of the lowest triangular filter in the
     * filter bank
     */
    virtual void setFMin(double f_min);
    /**
     * @brief Returns the frequency of the highest triangular filter in the
     * filter bank
     */
    virtual void setFMax(double f_max);
    /**
     * @brief Sets whether the frequencies of the filters in the filter bank
     * are taken from the linear or the Mel scale
     */
    virtual void setMelScale(bool mel_scale);
    /**
     * @brief Sets whether we used the energy or not
     */
    virtual void setEnergyFilter(bool energy_filter)
    { m_energy_filter = energy_filter; }


  protected:
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
    void filterBank(blitz::Array<double,1>& x);
    /**
     * @brief Applies the triangular filter bank to the input array and 
     * returns the logarithm of the magnitude in each band.
     */
    void logTriangularFilterBank(blitz::Array<double,1>& data) const;
    /**
     * @brief Applies the triangular filter bank to the input array and 
     * returns the magnitude in each band.
     */
    void triangularFilterBank(blitz::Array<double,1>& data) const;


    virtual void initWinLength();
    virtual void initWinSize();

    void initCacheHammingKernel();
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

    size_t m_n_filters;
    double m_f_min;
    double m_f_max;
    double m_pre_emphasis_coeff;
    bool m_mel_scale;
    double m_fb_out_floor;
    bool m_energy_filter;
    double m_log_fb_out_floor;

    blitz::Array<double,2> m_dct_kernel;
    blitz::Array<double,1> m_hamming_kernel;
    blitz::Array<int,1> m_p_index;
    std::vector<blitz::Array<double,1> > m_filter_bank;
    bob::sp::FFT1D m_fft;

    mutable blitz::Array<std::complex<double>,1> m_cache_frame_c1;
    mutable blitz::Array<std::complex<double>,1> m_cache_frame_c2;
    mutable blitz::Array<double,1> m_cache_filters;
};

}
}

#endif /* BOB_AP_SPECTROGRAM_H */
