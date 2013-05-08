/**
 * @file bob/ap/FrameExtractor.h
 * @date Wed Jan 11:10:20 2013 +0200
 * @author Elie Khoury <Elie.Khoury@idiap.ch>
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a rectangular window frame extractor
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

#ifndef BOB_AP_FRAME_EXTRACTOR_H
#define BOB_AP_FRAME_EXTRACTOR_H

#include <blitz/array.h>

namespace bob {
/**
 * \ingroup libap_api
 * @{
 *
 */
namespace ap {

/**
 * @brief This class is a base class for classes that perform audio processing
 * on a frame basis.
 */
class FrameExtractor
{
  public:
    /**
     * @brief Constructor. Initializes working arrays
     */
    FrameExtractor(const double sampling_frequency, 
      const double win_length_ms=20., const double win_shift_ms=10.);

    /**
     * @brief Copy Constructor
     */
    FrameExtractor(const FrameExtractor& other); 

    /**
     * @brief Assignment operator
     */
    FrameExtractor& operator=(const FrameExtractor& other);

    /**
     * @brief Equal to
     */
    bool operator==(const FrameExtractor& other) const;

    /**
     * @brief Not equal to
     */
    bool operator!=(const FrameExtractor& other) const;

    /**
     * @brief Destructor
     */
    virtual ~FrameExtractor();

    /**
     * @brief Gets the output shape for a given input/input length
     */
    virtual blitz::TinyVector<int,2> getShape(const size_t input_length) const;
    virtual blitz::TinyVector<int,2> getShape(const blitz::Array<double,1>& input) const;

    /**
     * @brief Returns the sampling frequency/frequency rate
     */
    double getSamplingFrequency() const
    { return m_sampling_frequency; }
    /**
     * @brief Returns the window length in miliseconds
     */
    double getWinLengthMs() const
    { return m_win_length_ms; }
    /**
     * @brief Returns the window length in number of samples
     */
    size_t getWinLength() const
    { return m_win_length; }
    /**
     * @brief Returns the window shift in miliseconds
     */
    double getWinShiftMs() const
    { return m_win_shift_ms; }
    /**
     * @brief Returns the window shift in number of samples
     */
    size_t getWinShift() const
    { return m_win_shift; }

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

  protected:
    /**
     * @brief Extracts the frame of the given index
     * @warning No check is performed
     */
    virtual void extractNormalizeFrame(const blitz::Array<double,1>& input, 
      const size_t i, blitz::Array<double,1>& frame) const;
    virtual void initWinSize();
    virtual void initWinLength();
    virtual void initWinShift();

    double m_sampling_frequency; ///< The sampling frequency
    double m_win_length_ms; ///< The window length in miliseconds 
    size_t m_win_length;
    double m_win_shift_ms;
    size_t m_win_shift;
    size_t m_win_size;

    mutable blitz::Array<double,1> m_cache_frame_d;
};

}
}

#endif /* BOB_AP_FRAME_EXTRACTOR_H */
