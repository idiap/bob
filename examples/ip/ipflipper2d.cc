#include "ipIntegral.h"
#include "Tensor.h"
#include <cassert>
#include "ipFlip.h"

using namespace Torch;

//////////////////////////////////////////////////////////////////////////////////
// MAIN
//////////////////////////////////////////////////////////////////////////////////

int main()
{
	print("new ShortTensor ...\n");
	// ShortTensor *st = new ShortTensor(5, 5);
	ShortTensor st(10, 15);
	st.fill(0);
	st.set(1, 1, 2);
	st.set(1, 2, 3);
	st.set(1, 3, 4);

	st.set(1, 1, 7);
	st.set(2, 1, 8);
	st.set(3, 1, 9);

	st.print("original");

	ipFlip ipflipper;

	assert(ipflipper.setBOption("vertical", true) == true);
	ipflipper.process(st);
	ipflipper.getOutput(0).print("flipped over vertical");

	assert(ipflipper.setBOption("vertical", false) == true);
	ipflipper.process(st);
	ipflipper.getOutput(0).print("flipped over horizontal");

	return 0;

}

