#include "torch5spro.h"

using namespace Torch;

///////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////

char *strFilename(char *filename, char dirsep = '/');
char *strBasename(char *filename, char extsep = '.');
void drawDetections(Image& image, const PatternList& detections, const Color& color);
void drawDetection(Image& image, const Pattern& detection, const Color& color);

int main(int argc, char* argv[])
{
	char* in_videoname = 0;
	char* in_paramname = 0;
	char* out_videoname = 0;
	bool verbose = false;

	// Read the command line
	CmdLine cmd;
	cmd.info("Face detection program to process a video frame by frame.\n");
	cmd.addSCmdArg("in_video", &in_videoname, "input video");
	cmd.addSCmdArg("in_params", &in_paramname, "face detector parameters file");
	cmd.addSCmdArg("out_video", &out_videoname, "output video with face detections drawn for each frame");
	
	cmd.addText("Options:");
	
	cmd.addBCmdOption("-verbose", &verbose, false, "verbose");

	if (cmd.read(argc, argv) < 0)
	{
		return 0;
	}

	// Load the face finder
	FaceFinder ffinder;
	CHECK_FATAL(ffinder.reset(in_paramname) == true);
	
	Scanner& scanner = ffinder.getScanner();
	TrackContextExplorer* ctx_explorer = dynamic_cast<TrackContextExplorer*>(&scanner.getExplorer());
	if (ctx_explorer == 0)
	{
		print("Error: need a TrackContextExplorer object!\n");
		return 1;
	}

	// Open the input/output videos for reading/writing
	Video in_video, out_video;
	CHECK_FATAL(in_video.setBOption("verbose", verbose) == true);
	CHECK_FATAL(in_video.open(in_videoname, "r") == true);
	
	CHECK_FATAL(out_video.setBOption("verbose", verbose) == true);
	CHECK_FATAL(out_video.setIOption("width", in_video.getIOption("width")) == true);
	CHECK_FATAL(out_video.setIOption("height", in_video.getIOption("height")) == true);
	CHECK_FATAL(out_video.setFOption("bitrate", in_video.getFOption("bitrate")) == true);
	CHECK_FATAL(out_video.setFOption("framerate", in_video.getFOption("framerate")) == true);
	CHECK_FATAL(out_video.setIOption("gop", in_video.getIOption("gop")) == true);
	CHECK_FATAL(out_video.open(out_videoname, "w") == true);
			
	char *temp = strFilename(in_videoname);
	char *basename = strBasename(temp);
	
	int image_width = in_video.getIOption("width");
	int image_height = in_video.getIOption("height");

	print("Input video (%s): width=%d, height=%d, framerate=%f, n_images=%d, codec=[%s].\n",
	      		basename,
			image_width,
			image_height,
			in_video.getFOption("framerate"),
			in_video.getNFrames(),
			in_video.codec());

	// Load and process each frame from the input video
	Image image(1, 1, 1);

	int cnt_frames = 0, n_faces = 0, n_framesmultifaces = 0;
	bool firstface = false;
	PatternList prv_detections, crt_detections;

	while (in_video.read(image) == true)
	{
		cnt_frames ++;
		print(">>> reading [%d/%d] ...\r", cnt_frames, in_video.getNFrames());
		
		prv_detections.clear();
		prv_detections.add(crt_detections);
		
		// Reuse the detection in the previous frame to initialize the context-based model
		// (At the begining there is no previous detections so the full scanning will be done)
		ctx_explorer->setSeedPatterns(prv_detections);
		CHECK_FATAL(ffinder.process(image) == true);
		
		const PatternList& detections = ffinder.getPatterns();
		crt_detections.clear();
		crt_detections.add(detections);	
		
		const int n_detections = detections.size();
		if (n_detections >= 1)
		{
			firstface = true;
		}
		n_faces += n_detections;			  
		if(n_detections > 1) n_framesmultifaces++;		
		
		// Draw the detections and save the frame
		static Image save_image(1, 1, 3);
		save_image.resize(image.getWidth(), image.getHeight(), 3);
		save_image.copyFrom(image);			
		if (firstface == true)
		{
			drawDetections(save_image, prv_detections, red);
			drawDetections(save_image, crt_detections, green);
		}		
		CHECK_FATAL(out_video.write(save_image) == true);
	}

	if (verbose == true)
	{
		print("Real number of frames: %d\n", cnt_frames);
	}
	//CHECK_FATAL(cnt_frames == in_video.getNFrames());

	// Close the videos
	in_video.close();
	out_video.close();
	
	print("\nOK\n");
	print("Output video (%s): nframes=%d nfaces=%d nframesmultifaces=%d \n", 
	      basename, cnt_frames, n_faces, n_framesmultifaces);
	print("\n");

	delete [] basename;

	return 0;
}

char *strFilename(char *filename, char dirsep) 
{
	char *p = strrchr(filename, dirsep);
	return p ? (p+1) : filename;
}

char *strBasename(char *filename, char extsep)
{
	char *copy = NULL;
	int len = strlen(filename);
	char *p = filename + len - 1;
	int i=len-1;
	while (*p != extsep && i-- >0) p--;
	if (i>0) 
	{
		copy = new char [i+1];
		strncpy(copy,filename,i);
		copy[i] = '\0';
	} 
	else 
	{
		copy = new char [len+1];
		strcpy(copy,filename);
	}
	return copy;
}

void drawDetections(Image& image, const PatternList& detections, const Color& color)
{
	const int n_detections = detections.size();
	for (int i = 0; i < n_detections; i ++)
	{
		drawDetection(image, detections.get(i), color);
	}
}

void drawDetection(Image& image, const Pattern& detection, const Color& color)
{
	image.drawRect(detection.m_x, detection.m_y, detection.m_w, detection.m_h, color);
	image.drawRect(detection.m_x - 1, detection.m_y - 1, detection.m_w + 2, detection.m_h + 2, color);
	image.drawRect(detection.m_x + 1, detection.m_y + 1, detection.m_w - 2, detection.m_h - 2, color);
}
