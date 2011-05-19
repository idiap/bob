#ifndef MACHINE_H
#define MACHINE_H
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
     */
    virtual void forward(const T_input& input, T_output& output) const = 0;
};


}
}
#endif // MACHINE_H
