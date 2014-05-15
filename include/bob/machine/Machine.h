/**
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_MACHINE_MACHINE_H
#define BOB_MACHINE_MACHINE_H

#include <cstring>

/**
 * @addtogroup MACHINE machine
 * @brief Machine module API
 */
namespace bob {
/**
 * @ingroup MACHINE
 */
namespace machine {
/**
 * @ingroup MACHINE
 * @{
 */

/**
 * Root class for all machines
 */
template<class T_input, class T_output>
class Machine
{
  public:
    virtual ~Machine() {}

    /**
     * Execute the machine
     *
     * @param input input data used by the machine
     * @param output value computed by the machine
     * @warning Inputs are checked
     */
    virtual void forward(const T_input& input, T_output& output) const = 0;

    /**
     * Execute the machine
     *
     * @param input input data used by the machine
     * @param output value computed by the machine
     * @warning Inputs are NOT checked
     */
    virtual void forward_(const T_input& input, T_output& output) const = 0;
};

/**
 * @}
 */
}}
#endif 
