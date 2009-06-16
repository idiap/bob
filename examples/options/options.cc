#include "torch5spro.h"

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
		CHECK_FATAL(object.addBOption(str, false) == true);
		CHECK_FATAL(object.addBOption(str, false) == false);

		sprintf(str, "Ioption%d", i);
		CHECK_FATAL(object.addIOption(str, i) == true);
		CHECK_FATAL(object.addIOption(str, i) == false);

		sprintf(str, "Foption%d", i);
		CHECK_FATAL(object.addFOption(str, i + 0.0f) == true);
		CHECK_FATAL(object.addFOption(str, i + 0.0f) == false);

		sprintf(str, "Doption%d", i);
		CHECK_FATAL(object.addDOption(str, i + 0.0) == true);
		CHECK_FATAL(object.addDOption(str, i + 0.0) == false);

		bool ok;

		// Retrieve and try to change the values
		sprintf(str, "Boption%d", i);
		CHECK_FATAL(object.getBOption(str) == false);
		CHECK_FATAL(object.getBOption(str, &ok) == false && ok == true);
		object.getBOption("dasdasd", &ok);
		CHECK_FATAL(ok == false);
		CHECK_FATAL(object.setBOption(str, true) == true);
		CHECK_FATAL(object.getBOption(str, &ok) == true && ok == true);

		sprintf(str, "Ioption%d", i);
		CHECK_FATAL(object.getIOption(str) == i);
		CHECK_FATAL(object.getIOption(str, &ok) == i && ok == true);
		object.getIOption("dasdasd", &ok);
		CHECK_FATAL(ok == false);
		CHECK_FATAL(object.setIOption(str, i + 1) == true);
		CHECK_FATAL(object.getIOption(str, &ok) == i + 1 && ok == true);

		sprintf(str, "Foption%d", i);
		CHECK_FATAL(object.getFOption(str) == i + 0.0f);
		CHECK_FATAL(object.getFOption(str, &ok) == i + 0.0f && ok == true);
		object.getFOption("dasdasd", &ok);
		CHECK_FATAL(ok == false);
		CHECK_FATAL(object.setFOption(str, i + 1.0f) == true);
		CHECK_FATAL(object.getFOption(str, &ok) == i + 1.0f && ok == true);

		sprintf(str, "Doption%d", i);
		CHECK_FATAL(object.getDOption(str) == i + 0.0);
		CHECK_FATAL(object.getDOption(str, &ok) == i + 0.0 && ok == true);
		object.getDOption("dasdasd", &ok);
		CHECK_FATAL(ok == false);
		CHECK_FATAL(object.setDOption(str, i + 1.0) == true);
		CHECK_FATAL(object.getDOption(str, &ok) == i + 1.0 && ok == true);

		print("PASSED test [%d/%d]\r", i + 1, n_tests);
	}

	print("\nOK\n");

	return 0;
}

