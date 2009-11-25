#include "torch5spro.h"

using namespace Torch;

///////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
	char* in_videoname = 0;
	bool verbose = false;
	bool check_header_only = false;
	int check_w = -1;
	int check_h = -1;
	float check_fps = -1;

	// Read the command line
	CmdLine cmd;
	cmd.info("Program to check videos.\n");
	cmd.addSCmdArg("in_video", &in_videoname, "input video");

	cmd.addText("Options:");
	cmd.addBCmdOption("-verbose", &verbose, false, "verbose");
	cmd.addBCmdOption("-cheader", &check_header_only, false, "check the header only");
	cmd.addICmdOption("-cw", &check_w, -1, "check width");
	cmd.addICmdOption("-ch", &check_h, -1, "check height");
	cmd.addFCmdOption("-cf", &check_fps, -1, "check framerate");

	cmd.read(argc, argv);

	// Open the input video for reading
	Video in_video;
	in_video.setBOption("verbose", verbose);

	CHECK_FATAL(in_video.open(in_videoname, "r") == true);
	print("Input video: [%dx%d], %f bitrate, %f framerate, %d gop, %d frames [%s].\n",
		in_video.getIOption("width"),
		in_video.getIOption("height"),
		in_video.getFOption("bitrate"),
		in_video.getFOption("framerate"),
		in_video.getIOption("gop"),
		in_video.getNFrames(),
		in_video.codec());

	if(check_w != -1 && check_w != in_video.getIOption("width"))
	{
		warning("Incorrect width (%d <> %d) in the video file %s", check_w, in_video.getIOption("width"), in_videoname);

		in_video.close();

		return 1;
	}

	if(check_h != -1 && check_h != in_video.getIOption("height"))
	{
		warning("Incorrect height (%d <> %d) in the video file %s", check_h, in_video.getIOption("height"), in_videoname);

		in_video.close();

		return 1;
	}

	if(check_fps != -1 && check_fps != in_video.getFOption("framerate"))
	{
		warning("Incorrect framerate (%g <> %g) in the video file %s", check_fps, in_video.getFOption("framerate"), in_videoname);

		in_video.close();

		return 1;
	}

	if(check_header_only)
	{
		in_video.close();

		print("\nOK\n");

		return 0;
	}

	// Load each frame from the input video, save it and write it to the output video
	Image image(1, 1, 3);	// This will read RGB images!!! (no matter what the input video frames)
	//Image image(1, 1, 1);	// This will read grayscale images!!! (no matter what the input video frames)

	int cnt_frames = 0;
	while (in_video.read(image) == true)
	{
		cnt_frames ++;
		print(">>> reading [%d/%d] ...\r", cnt_frames, in_video.getNFrames());

	}

	print("Real number of frames: %d\n", cnt_frames);
	if(cnt_frames != in_video.getNFrames()) 
		warning("Incorrect number of frames in the video file %s", in_videoname);
	   
	//CHECK_FATAL(cnt_frames == in_video.getNFrames());

	// Close the videos
	in_video.close();

	print("\nOK\n");

	return 0;
}

