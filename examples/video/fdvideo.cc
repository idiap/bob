#include "torch5spro.h"

using namespace Torch;

///////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
	char* in_videoname = 0;
	char* in_paramname = 0;
	char* out_videoname = 0;
	char* out_resname;
	bool draw = false;
	bool verbose = false;

	// Read the command line
	CmdLine cmd;
	cmd.info("Face detection program to process a video frame by frame.\n");
	cmd.addSCmdArg("in_video", &in_videoname, "input video");
	cmd.addSCmdArg("in_params", &in_paramname, "face detector parameters file");
	cmd.addSCmdArg("out_results", &out_resname, "output face detection results file");

	cmd.addText("Options:");
	cmd.addSCmdOption("-out_video", &out_videoname, "", "output video with face detections drawn for each frame");
	cmd.addBCmdOption("-draw", &draw, false, "draw face detections for each frame");
	cmd.addBCmdOption("-verbose", &verbose, false, "verbose");

	cmd.read(argc, argv);

	// Load the face finder
	FaceFinder ffinder;
	CHECK_FATAL(ffinder.reset(in_paramname) == true);

	// Open the result file
	File out_file;
	CHECK_FATAL(out_file.open(out_resname, "w") == true);

	// Open the input/output videos for reading/writing
	Video in_video, out_video;
	CHECK_FATAL(in_video.setBOption("verbose", verbose) == true);
	CHECK_FATAL(in_video.open(in_videoname, "r") == true);
	if (draw == true)
	{
		CHECK_FATAL(out_video.setBOption("verbose", verbose) == true);
		CHECK_FATAL(out_video.setIOption("width", in_video.getIOption("width")) == true);
		CHECK_FATAL(out_video.setIOption("height", in_video.getIOption("height")) == true);
		CHECK_FATAL(out_video.setFOption("bitrate", in_video.getFOption("bitrate")) == true);
		CHECK_FATAL(out_video.setFOption("framerate", in_video.getFOption("framerate")) == true);
		CHECK_FATAL(out_video.setIOption("gop", in_video.getIOption("gop")) == true);

		CHECK_FATAL(out_video.open(out_videoname, "w") == true);
	}

	// Load and process each frame from the input video
	Image image(1, 1, 1);

	int cnt_frames = 0;
	while (in_video.read(image) == true)
	{
		cnt_frames ++;
		print(">>> reading [%d/%d] ...\r", cnt_frames, in_video.getNFrames());

		// Process the frame
		CHECK_FATAL(ffinder.process(image) == true);
		const PatternList& detections = ffinder.getPatterns();
		const int n_detections = detections.size();

		// Save the detections to the result file
		out_file.printf("%d %d\n", cnt_frames, n_detections);
		for (int i = 0; i < n_detections; i ++)
		{
			const Pattern& det = detections.get(i);
			out_file.printf("%d %d %d %d %lf\n", det.m_x, det.m_y, det.m_w, det.m_h, det.m_confidence);
		}

		// Draw the detections (if requested) and save the fram
		if (draw == true)
		{
			static Image save_image(1, 1, 3);
			save_image.resize(image.getWidth(), image.getHeight(), 3);
			save_image.copyFrom(image);

			for (int i = 0; i < n_detections; i ++)
			{
				const Pattern& det = detections.get(i);
				save_image.drawRect(det.m_x, det.m_y, det.m_w, det.m_h, red);
				save_image.drawRect(det.m_x - 1, det.m_y - 1, det.m_w + 2, det.m_h + 2, red);
				save_image.drawRect(det.m_x + 1, det.m_y + 1, det.m_w - 2, det.m_h - 2, red);
			}

			CHECK_FATAL(out_video.write(save_image) == true);
		}
	}

	print("Real number of frames: %d\n", cnt_frames);

	//CHECK_FATAL(cnt_frames == in_video.getNFrames());

	// Close the videos and the result file
	in_video.close();
	if (draw == true)
	{
		out_video.close();
	}
	out_file.close();

	return 0;
}

