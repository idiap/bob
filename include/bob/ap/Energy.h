/**
 * @author Elie Khoury <Elie.Khoury@idiap.ch>
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Wed Jan 11:10:20 2013 +0200
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */


#ifndef BOB_AP_ENERGY_H
#define BOB_AP_ENERGY_H

#include <blitz/array.h>
#include "FrameExtractor.h"

namespace bob {
/**
 * \ingroup libap_api
 * @{
 *
 */
namespace ap {

/**
 * @brief This class allows the extraction of energy in a frame basis
 */
class Energy: public FrameExtractor
{
  public:
    /**
     * @brief Constructor. Initializes working arrays
     */
    Energy(const double sampling_frequency, const double win_length_ms=20.,
      const double win_shift_ms=10.);

    /**
     * @brief Copy constructor
     */
    Energy(const Energy& other);

    /**
     * @brief Assignment operator
     */
    Energy& operator=(const Energy& other);

    /**
     * @brief Equal to
     */
    bool operator==(const Energy& other) const;

    /**
     * @brief Not equal to
     */
    bool operator!=(const Energy& other) const;

    /**
     * @brief Destructor
     */
    virtual ~Energy();

    /**
     * @brief Gets the Energy features shape for a given input/input length
     */
    virtual blitz::TinyVector<int,2> getShape(const size_t input_length) const;
    virtual blitz::TinyVector<int,2> getShape(const blitz::Array<double,1>& input) const;

    /**
     * @brief Computes Energy features
     */
    void operator()(const blitz::Array<double,1>& input, blitz::Array<double,1>& output);

    /**
     * @brief Gets the energy floor
     */
    virtual double getEnergyFloor() const
    { return m_energy_floor; }

    /**
     * @brief Sets the energy floor
     */
    virtual void setEnergyFloor(double energy_floor)
    { m_energy_floor = energy_floor;
      m_log_energy_floor = log(m_energy_floor); }

  protected:
    /**
     * @brief Computes the logarithm of the energy
     */
    double logEnergy(blitz::Array<double,1> &data) const;

    double m_energy_floor;
    double m_log_energy_floor;
};

}}

#endif /* BOB_AP_ENERGY_H */
