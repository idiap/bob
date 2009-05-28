#include "Image.h"
#include "ipHisto.h"
#include "ipHaarLienhart.h"
#include "ipFlip.h"

using namespace Torch;

///////////////////////////////////////////////////////////////////////////
// Tensor test: allocate a video
///////////////////////////////////////////////////////////////////////////

void test_video()
{
	// Allocate a video (frames x nplanes)
	const int n_frames = 128 + rand() % 512;
	const int n_planes = 1 + rand() % 3;
	print("Video: allocating [%d] frames x [%d] planes ...\n", n_frames, n_planes);

	Image*** video = new Image**[n_frames];
	manage_array(video);
	for (int i = 0; i < n_frames; i ++)
	{
		video[i] = manage_array(new Image*[n_planes]);
		for (int j = 0; j < n_planes; j ++)
		{
			video[i][j] = manage(new Image(128, 128, 1));
		}
	}

	// Try to manage the same object (it should work)
	print("Video: manage again the same video ...\n");

	video = manage_array(video);
	for (int i = 0; i < n_frames; i ++)
	{
		video[i] = manage_array(video[i]);
		for (int j = 0; j < n_planes; j ++)
		{
			video[i][j] = manage(video[i][j]);
		}
	}

	// Unmanage the main object
	print("Video: unmanage the main pointer and manually delete it ...\n");

	unmanage(video);
	manage_array(video);
	unmanage(video);
	delete[] video;

	print("\nOK\n\n");
}

///////////////////////////////////////////////////////////////////////////
// Object test: allocate a battery of ipCores
///////////////////////////////////////////////////////////////////////////

void test_ipCores()
{
	// Allocate the ipCores
	const int size1 = 2 + rand() % 5;
	const int size2 = 2 + rand() % 5;
	print("IpCores: allocating [%d] x [%d] ...\n", size1, size2);

	ipCore*** cores = manage_array(new ipCore**[size1]);
	for (int i = 0; i < size1; i ++)
	{
		cores[i] = manage_array(new ipCore*[size2]);
		for (int j = 0; j < size2; j ++)
		{
			switch(rand() % 3)
			{
			case 0:
				cores[i][j] = new ipHaarLienhart;
				break;

			case 1:
				cores[i][j] = new ipHisto;
				break;

			case 2:
				cores[i][j] = new ipFlip;
				break;

			default:
				break;
			}

			manage(cores[i][j]);
		}
	}

	// Try to manage the same object (it should work)
	print("IpCores: manage again the same ipCores ...\n");

	cores = manage_array(cores);
	for (int i = 0; i < size1; i ++)
	{
		cores[i] = manage_array(cores[i]);
		for (int j = 0; j < size2; j ++)
		{
			cores[i][j] = manage(cores[i][j]);
		}
	}

	// Unmanage the main object
	print("IpCores: unmanage the main pointer and manually delete it ...\n");

	unmanage(cores);
	manage_array(cores);
	unmanage(cores);
	delete[] cores;

	print("\nOK\n\n");
}

///////////////////////////////////////////////////////////////////////////
// Object test: allocate a battery of builtin types
///////////////////////////////////////////////////////////////////////////

void test_builtin()
{
	print("Builtin: testing ...\n");

	manage(new float(1.0f));
	manage(new int);
	manage(new double);

	manage_array(new int*[3]);
	manage_array(new int[4]);
	manage_array(new float*[3]);
	manage_array(new float[4]);
	manage_array(new double*[3]);
	manage_array(new double[4]);

	manage_array(new int**[23]);

	print("\nOK\n\n");
}

///////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
	srand((unsigned int)time(0));

	test_video();
	test_ipCores();
	test_builtin();

	return 0;
}

