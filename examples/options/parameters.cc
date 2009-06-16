#include "torch5spro.h"

using namespace Torch;

int main()
{
	Parameters p;

	unsigned long seed = THRandom_seed();

	// Do lots of tests ...
	const int n_tests = 1;
	float random_;
	for (int t = 0; t < n_tests; t++)
	{
	   	int i = 4; //t;

		char str[200];

		// Add
		sprintf(str, "I%d", i);
		CHECK_FATAL(p.addI(str, i) == true);
		CHECK_FATAL(p.addI(str, i) == false);

		sprintf(str, "F%d", i);
		CHECK_FATAL(p.addF(str, i + 0.0f) == true);
		CHECK_FATAL(p.addF(str, i + 0.0f) == false);

		sprintf(str, "D%d", i);
		CHECK_FATAL(p.addD(str, i + 0) == true);
		CHECK_FATAL(p.addD(str, i + 0) == false);

		sprintf(str, "I*%d", i);
		CHECK_FATAL(p.addIarray(str, i+1, i) == true);
		CHECK_FATAL(p.addIarray(str, i+1, i) == false);

		sprintf(str, "F*%d", i);
		CHECK_FATAL(p.addFarray(str, i+1, i + 0.0f) == true);
		CHECK_FATAL(p.addFarray(str, i+1, i + 0.0f) == false);

		sprintf(str, "D*%d", i);
		CHECK_FATAL(p.addDarray(str, i+1, i + 0) == true);
		CHECK_FATAL(p.addDarray(str, i+1, i + 0) == false);
	}

	p.print("Saved parameters");

	File ofile;
	ofile.open("test.params", "w");
	p.saveFile(ofile);
	ofile.close();

	for (int t = 0; t < n_tests; t++)
	{
	   	int i = 4; //t;

		char str[200];

		bool ok;

		// Retrieve and try to change the values
		sprintf(str, "I%d", i);
		CHECK_FATAL(p.getI(str) == i);
		CHECK_FATAL(p.getI(str, &ok) == i && ok == true);
		p.getI("dasdasd", &ok);
		CHECK_FATAL(ok == false);
		CHECK_FATAL(p.setI(str, i + 1) == true);
		CHECK_FATAL(p.getI(str, &ok) == i + 1 && ok == true);

		sprintf(str, "F%d", i);
		CHECK_FATAL(p.getF(str) == i + 0.0f);
		CHECK_FATAL(p.getF(str, &ok) == i + 0.0f && ok == true);
		p.getF("dasdasd", &ok);
		CHECK_FATAL(ok == false);
		CHECK_FATAL(p.setF(str, i + 1.0f) == true);
		CHECK_FATAL(p.getF(str, &ok) == i + 1.0f && ok == true);

		sprintf(str, "D%d", i);
		CHECK_FATAL(p.getD(str) == i + 0.0);
		CHECK_FATAL(p.getD(str, &ok) == i + 0.0 && ok == true);
		p.getD("dasdasd", &ok);
		CHECK_FATAL(ok == false);
		CHECK_FATAL(p.setD(str, i + 1.0) == true);
		CHECK_FATAL(p.getD(str, &ok) == i + 1.0 && ok == true);

		sprintf(str, "I*%d", i);
		int *iparams = p.getIarray(str);
		iparams = p.getIarray(str, &ok);
		CHECK_FATAL(ok == true);
		for(int j = 0 ; j < i+1 ; j++) iparams[j] = -1;

		sprintf(str, "F*%d", i);
		float *fparams = p.getFarray(str);
		fparams = p.getFarray(str, &ok);
		CHECK_FATAL(ok == true);
		for(int j = 0 ; j < i+1 ; j++) fparams[j] = -1;

		sprintf(str, "D*%d", i);
		double *dparams = p.getDarray(str);
		dparams = p.getDarray(str, &ok);
		CHECK_FATAL(ok == true);
		for(int j = 0 ; j < i+1 ; j++) dparams[j] = -1;

		print("PASSED test [%d/%d]\r", t + 1, n_tests);
	}

	p.print("Modified parameters");

	File ifile;
	ifile.open("test.params", "r");
	p.loadFile(ifile);
	ifile.close();

	p.print("Loaded parameters");


	Parameters q;
	for (int t = 0; t < n_tests; t++)
	{
	   	int i = 4; //t;

		char str[200];

		// Add
		sprintf(str, "I%d", i);
		CHECK_FATAL(q.addI(str, i) == true);

		sprintf(str, "F%d", i);
		CHECK_FATAL(q.addF(str, i + 0.0f) == true);

		sprintf(str, "D%d", i);
		CHECK_FATAL(q.addD(str, i + 0) == true);

		sprintf(str, "I*%d", i);
		CHECK_FATAL(q.addIarray(str) == true);

		sprintf(str, "F*%d", i);
		CHECK_FATAL(q.addFarray(str) == true);

		sprintf(str, "D*%d", i);
		CHECK_FATAL(q.addDarray(str) == true);
	}

	q.print("Parameters to load");

	File ifile2;
	ifile2.open("test.params", "r");
	q.loadFile(ifile2);
	ifile2.close();

	q.print("Re-Loaded parameters");

	print("\nOK\n");

	return 0;
}

