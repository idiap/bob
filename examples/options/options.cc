#include "Object.h"
#include <cassert>

using namespace Torch;

int main()
{
	Object object;

	// Do lots of tests ...
	const int n_tests = 1000;
	for (int i = 0; i < n_tests; i ++)
	{
		char str[200];

		// Add options
		sprintf(str, "Boption%d", i);
		assert(object.addBOption(str, false) == true);
		assert(object.addBOption(str, false) == false);

		sprintf(str, "Ioption%d", i);
		assert(object.addIOption(str, i) == true);
		assert(object.addIOption(str, i) == false);

		sprintf(str, "Foption%d", i);
		assert(object.addFOption(str, i + 0.0f) == true);
		assert(object.addFOption(str, i + 0.0f) == false);

		sprintf(str, "Doption%d", i);
		assert(object.addDOption(str, i + 0.0) == true);
		assert(object.addDOption(str, i + 0.0) == false);

		bool ok;

		// Retrieve and try to change the values
		sprintf(str, "Boption%d", i);
		assert(object.getBOption(str) == false);
		assert(object.getBOption(str, &ok) == false && ok == true);
		object.getBOption("dasdasd", &ok);
		assert(ok == false);
		assert(object.setBOption(str, true) == true);
		assert(object.getBOption(str, &ok) == true && ok == true);

		sprintf(str, "Ioption%d", i);
		assert(object.getIOption(str) == i);
		assert(object.getIOption(str, &ok) == i && ok == true);
		object.getIOption("dasdasd", &ok);
		assert(ok == false);
		assert(object.setIOption(str, i + 1) == true);
		assert(object.getIOption(str, &ok) == i + 1 && ok == true);

		sprintf(str, "Foption%d", i);
		assert(object.getFOption(str) == i + 0.0f);
		assert(object.getFOption(str, &ok) == i + 0.0f && ok == true);
		object.getFOption("dasdasd", &ok);
		assert(ok == false);
		assert(object.setFOption(str, i + 1.0f) == true);
		assert(object.getFOption(str, &ok) == i + 1.0f && ok == true);

		sprintf(str, "Doption%d", i);
		assert(object.getDOption(str) == i + 0.0);
		assert(object.getDOption(str, &ok) == i + 0.0 && ok == true);
		object.getDOption("dasdasd", &ok);
		assert(ok == false);
		assert(object.setDOption(str, i + 1.0) == true);
		assert(object.getDOption(str, &ok) == i + 1.0 && ok == true);

		print("PASSED test [%d/%d]\r", i + 1, n_tests);
	}

	print("\nOK\n");

	return 0;
}

