#ifndef TORCH5SPRO_MACHINE_MACHINE_H
#define TORCH5SPRO_MACHINE_MACHINE_H
#include <cstring>

namespace Torch {
namespace machine {


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


}
}
#endif // TORCH5SPRO_MACHINE_MACHINE_H
