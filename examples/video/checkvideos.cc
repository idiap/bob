#include "torch5spro.h"

using namespace Torch;

///////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
	bool verbose = false;
	bool check_header_only = false;
	int check_w = -1;
	int check_h = -1;
	float check_fps = -1;

	FileListCmdOption* video_files = new FileListCmdOption("videos", "video files");
	video_files->isArgument(true);


	// Read the command line
	CmdLine cmd;
	cmd.info("Program to check videos.\n");
	cmd.addCmdOption(video_files);

	cmd.addText("Options:");
	cmd.addBCmdOption("-verbose", &verbose, false, "verbose");
	cmd.addBCmdOption("-cheader", &check_header_only, false, "check the header only");
	cmd.addICmdOption("-cw", &check_w, -1, "check width");
	cmd.addICmdOption("-ch", &check_h, -1, "check height");
	cmd.addFCmdOption("-cf", &check_fps, -1, "check framerate");

	cmd.read(argc, argv);

	if (video_files->n_files < 1)
	{
		print("Error: no video file provided!\n");
		return 1;
	}

	// Open the input video for reading
	Video in_video;
	in_video.setBOption("verbose", verbose);

	int n_non_existing_files = 0;
	int n_incorrect_size = 0;
	int n_incorrect_framerate = 0;
	int n_incorrect_nframes = 0;

	print("Number of video files to check: %d\n", video_files->n_files);
	
	for(int i = 0 ; i < video_files->n_files ; i++)
	{
		if(in_video.open(video_files->file_names[i], "r") == false)
		{
			warning("Impossible to open the video file %s", video_files->file_names[i]);

			n_non_existing_files++;
		}
		else
		{
			print("Input video [%d/%d] (%s) ...\n", i+1, video_files->n_files, video_files->file_names[i]);

			print("   [%dx%d], %f bitrate, %f framerate, %d gop, %d frames [%s].\n",
				in_video.getIOption("width"),
				in_video.getIOption("height"),
				in_video.getFOption("bitrate"),
				in_video.getFOption("framerate"),
				in_video.getIOption("gop"),
				in_video.getNFrames(),
				in_video.codec());

		bool incorrect_size = false;
		if(check_w != -1 && check_w != in_video.getIOption("width"))
		{
			warning("Incorrect width (%d <> %d) in the video file %s", check_w, in_video.getIOption("width"), video_files->file_names[i]);

			incorrect_size = true;
		}

		if(check_h != -1 && check_h != in_video.getIOption("height"))
		{
			warning("Incorrect height (%d <> %d) in the video file %s", check_h, in_video.getIOption("height"), video_files->file_names[i]);

			incorrect_size = true;
		}

		if(incorrect_size) n_incorrect_size++;

		if(check_fps != -1 && check_fps != in_video.getFOption("framerate"))
		{
			warning("Incorrect framerate (%g <> %g) in the video file %s", check_fps, in_video.getFOption("framerate"), video_files->file_names[i]);

			n_incorrect_framerate++;
		}

		if(check_header_only == false)
		{
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
			{
				warning("Incorrect number of frames in the video file %s", video_files->file_names[i]);
				n_incorrect_nframes++;
			}  
		}

		// Close the videos
		in_video.close();

		print("OK\n");
		}
	}

	print("Stats:\n");
	print(" n_non_existing_files = %d\n", n_non_existing_files);
	print(" n_incorrect_size = %d\n", n_incorrect_size);
	print(" n_incorrect_framerate = %d\n", n_incorrect_framerate);
	print(" n_incorrect_nframes = %d\n", n_incorrect_nframes);

	return 0;
}

