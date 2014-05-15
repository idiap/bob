/**
 * @author Elie Khoury <Elie.Khoury@idiap.ch>
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Wed Jan 11:10:20 2013 +0200
 *
 * @brief Implement Linear and Mel Frequency Cepstral Coefficients
 * functions (MFCC and LFCC)
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */


#ifndef BOB_AP_CEPS_H
#define BOB_AP_CEPS_H

#include <blitz/array.h>
#include "Spectrogram.h"

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
//class CepsTest;

/**
 * @brief This class allows the extraction of features from raw audio data.
 * References:
 *  1. SPro tools (http://www.irisa.fr/metiss/guig/spro/spro-4.0.1/spro.html)
 *  2. Wikipedia (http://en.wikipedia.org/wiki/Mel-frequency_cepstrum).
 */
class Ceps: public Spectrogram
{
  public:
    /**
     * @brief Constructor. Initializes working arrays
     */
    Ceps(const double sampling_frequency, const double win_length_ms=20.,
      const double win_shift_ms=10., const size_t n_filters=24,
      const size_t n_ceps=19, const double f_min=0.,
      const double f_max=4000., const size_t delta_win=2,
      const double pre_emphasis_coef=0.95, const bool mel_scale=true,
      const bool dct_norm=false);

    /**
     * @brief Copy constructor.
     */
    Ceps(const Ceps& other);

    /**
     * @brief Assignment operator
     */
    Ceps& operator=(const Ceps& other);

    /**
     * @brief Equal to
     */
    bool operator==(const Ceps& other) const;

    /**
     * @brief Not equal to
     */
    bool operator!=(const Ceps& other) const;

    /**
     * @brief Destructor
     */
    virtual ~Ceps();

    /**
     * @brief Gets the Cepstral features shape for a given input/input length
     */
    blitz::TinyVector<int,2> getShape(const size_t input_length) const;
    blitz::TinyVector<int,2> getShape(const blitz::Array<double,1>& input) const;

    /**
     * @brief Computes Cepstral features
     */
    void operator()(const blitz::Array<double,1>& input, blitz::Array<double,2>& output);

    /**
     * @brief Returns the sampling frequency/frequency rate
     */
    double getSamplingFrequency() const
    { return m_sampling_frequency; }
    /**
     * @brief Returns the number of cepstral coefficient to keep
     */
    size_t getNCeps() const
    { return m_n_ceps; }
    /**
     * @brief Rerturns the size of the window used to compute first and second
     * order derivatives
     */
    size_t getDeltaWin() const
    { return m_delta_win; }
    /**
     * @brief Tells whether the DCT coefficients are normalized or not
     */
    bool getDctNorm() const
    { return m_dct_norm; }
    /**
     * @brief Tells whether the energy is added to the cepstral coefficients
     * or not
     */
    bool getWithEnergy() const
    { return m_with_energy; }
    /**
     * @brief Tells whether the first order derivatives are added to the
     * cepstral coefficients or not
     */
    bool getWithDelta() const
    { return m_with_delta; }
    /**
     * @brief Tells whether the second order derivatives are added to the
     * cepstral coefficients or not
     */
    bool getWithDeltaDelta() const
    { return m_with_delta_delta; }

    /**
     * @brief Returns the number of filters to keep
     */
    virtual void setNFilters(size_t n_ceps);
    /**
     * @brief Returns the number of cepstral coefficient to keep
     */
    virtual void setNCeps(size_t n_ceps);
    /**
     * @brief Sets the size of the window used to compute first and second
     * order derivatives
     */
    virtual void setDeltaWin(size_t delta_win)
    { m_delta_win = delta_win; }
    /**
     * @brief Sets whether the DCT coefficients are normalized or not
     */
    virtual void setDctNorm(bool dct_norm);
    /**
     * @brief Sets whether the energy is added to the cepstral coefficients
     * or not
     */
    void setWithEnergy(bool with_energy)
    { m_with_energy = with_energy; }
    /**
     * @brief Sets whether the first order derivatives are added to the
     * cepstral coefficients or not
     */
    void setWithDelta(bool with_delta)
    { if (!with_delta) m_with_delta_delta = false;
      m_with_delta = with_delta; }
    /**
     * @brief Sets whether the first order derivatives are added to the
     * cepstral coefficients or not. If enabled, first order derivatives are
     * automatically enabled as well.
     */
    void setWithDeltaDelta(bool with_delta_delta)
    { if (with_delta_delta) m_with_delta = true;
      m_with_delta_delta = with_delta_delta; }

  private:
    /**
     * @brief Computes the first order derivative from the given input.
     * This methods is used to compute both the delta's and double delta's.
     */
    void addDerivative(const blitz::Array<double,2>& input, blitz::Array<double,2>& output) const;
    /**
     * @brief Applies the DCT to the cepstral features:
     * \f$out[i]=sqrt(2/N)*sum_{j=1}^{N} (in[j]cos(M_PI*i*(j-0.5)/N)\f$
     */
    void applyDct(blitz::Array<double,1>& ceps_row) const;

    void initCacheDctKernel();
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

    size_t m_n_ceps;
    size_t m_delta_win;
    bool m_dct_norm;
    bool m_with_energy;
    bool m_with_delta;
    bool m_with_delta_delta;

    blitz::Array<double,2> m_dct_kernel;

//    friend class TestCeps;
};
}}

#endif /* BOB_AP_CEPS_H */
