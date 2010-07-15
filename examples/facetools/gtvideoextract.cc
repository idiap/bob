// ALWAYS LAST
#include "torch5spro.h"

using namespace Torch;

// some very long constant strings
const char GT_FORMAT_INFO[] =
	"gt format (1=eyes center, 2=banca format, 3=eyes corners, \
4=eye corners + nose tip + chin, 5=left eye corners +		   \
right eye center + nose tip + chin, 6=left eye center +		   \
nose tip + chin, 7=Tim Cootes's markup 68 pts)";

const char ONE_GT_INFO[] =
	"if true then considers that the gt file contains one object, \
otherwise assumes that the first line of the file contains the	      \
number of objects";

// The main idea
int main(int argc, char *argv[])
{
	// handle the command line

	char *list_filename;
	char *image_pathname;
	char *gt_pathname;
	char *cfg_norm_filename;
	char *image_ext;
	char *gt_ext;
	int  gt_format;
	bool one_gt_object;
	bool verbose;
	bool oimage;
	bool onetensor;

	// Build the command line object
	CmdLine cmd;
	cmd.setBOption("write log", false);

	cmd.info("File List testing program");

	cmd.addText("\nArguments:");
	cmd.addSCmdArg("list_filename", &list_filename, "list of files");
	cmd.addSCmdArg("image_pathname", &image_pathname, "path to image files");
	cmd.addSCmdArg("gt_pathname", &gt_pathname, "path to gt files");
	cmd.addSCmdArg("configuration", &cfg_norm_filename, "input normalization configuration file");

	cmd.addText("\nOptions:");
	cmd.addSCmdOption("-image_ext", &image_ext, "avi", "image file extension");
	cmd.addSCmdOption("-gt_ext", &gt_ext, "pos", "gt file extension");
	cmd.addICmdOption("-gt_format", &gt_format, 1, GT_FORMAT_INFO);
	cmd.addBCmdOption("-one_gt_object", &one_gt_object, false, ONE_GT_INFO);
	cmd.addBCmdOption("-verbose", &verbose, false, "verbose");
	cmd.addBCmdOption("-oimage", &oimage, false, "output intermediate images");
	cmd.addBCmdOption("-onetensor", &onetensor, false, "generates only one tensor");

	// Parse the command line
	if (cmd.read(argc, argv) < 0) {
		return 1;
	}

	// handle all the files in the list

	FileList *file_list = new FileList(list_filename);

	// create space for the filenames

	char image_filename[BUFSIZ];
	char gt_filename[BUFSIZ];

	print("Number of files:%d\n", file_list->n_files);
	for(int i = 0 ; i < file_list->n_files ; i++) {

		// Piecing together where to find the videofile
		// and the ground truth file
		const char *fname = file_list->file_names[i];
		print("%s\n", fname);
		sprintf(image_filename,"%s/%s.%s", image_pathname, fname, image_ext);
		sprintf(gt_filename,   "%s/%s.%s", gt_pathname,    fname, gt_ext);

		// We will create an individual "onetensor" file for
		// each video
		TensorFile *onetensor_file = 0;
		char onetensor_filename[BUFSIZ];
		sprintf(onetensor_filename, "%s.onetensor.tensor", fname);

		// set the configurations for our normalizing object
		ipGeomNorm gnormalizer;
		CHECK_FATAL(gnormalizer.loadCfg(cfg_norm_filename) == true);

		// The object that holds the information needed
		// to normalize the frame.
		GTFile *gt_holder = new bbx2eye19x19deye10_GTFile();

		// The file that that holds the information about
		// all the frames. We will go through it in sequence
		File *file = new File();;
		file->open(gt_filename, "r");

		// start to read the video
		Video video;
		CHECK_FATAL(video.setBOption("verbose", 1) == true);
		CHECK_FATAL(video.open(image_filename, "r") == true);

		// for every frame in video
		Image current_frame(1,1,1);
		while ( video.read(current_frame) == true ) {

			// read in the "header" for each frame
			int frame;
			int faces;
			file->scanf(" %d ", &frame);
			file->scanf(" %d ", &faces);

			// One frame can contain multiple faces, therefore
			// will we for every frame extract multiple faces
			// and normalize each face.
			for (int face = 0; face < faces; face++) {

				// let the gt_holder read up the information
				// from the file that it needs for the next
				// frame
				gt_holder->load(file);

				// the gt_holder will not read up the score,
				// therefore do it here, that is, read from
				// the file
				float score;
				file->scanf(" %g \n", &score);

				// we will now pass the information forward
				// to the object that is goin to the
				// normalization and read then normalize
				// the frame

				CHECK_FATAL(gnormalizer.setGTFile(gt_holder) == true);
				CHECK_FATAL(gnormalizer.process(current_frame) == true);

				const ShortTensor& norm_timage = (const ShortTensor&)gnormalizer.getOutput(0);

				print("Output image size:\n");
				print("   width   = %d\n", norm_timage.size(1));
				print("   height  = %d\n", norm_timage.size(0));

				///////////////////////////////////////////////
				// writing the frace to file
				///////////////////////////////////////////////

				// Convert the output color image
				// (3D tensor RGB) to a grayscale image (3D gray)
				Image imagegray(norm_timage.size(1), norm_timage.size(0), 1);

				// the cast is necessary other copyFrom will
				// not consider it as an image and will not
				// convert it to grayscal
				imagegray.copyFrom((Image &)norm_timage);

				if (onetensor_file == 0) {

					onetensor_file = new TensorFile;
					CHECK_FATAL(onetensor_file->openWrite(onetensor_filename,
									      Tensor::Short, 2,
									      norm_timage.size(0), norm_timage.size(1), 0, 0));

					const TensorFile::Header& onetensor_header = onetensor_file->getHeader();
					print("One tensor file:\n");
					print(" type:         [%s]\n", str_TensorTypeName[onetensor_header.m_type]);
					print(" n_dimensions: [%d]\n", onetensor_header.m_n_dimensions);
					print(" size[0]:      [%d]\n", onetensor_header.m_size[0]);
					print(" size[1]:      [%d]\n", onetensor_header.m_size[1]);
					print(" size[2]:      [%d]\n", onetensor_header.m_size[2]);
				}

				// Select the grayscale channel as a 2D tensor and save it !
				ShortTensor *t_ = new ShortTensor();
				t_->select(&imagegray, 2, 0);
				onetensor_file->save(*t_);
				delete t_;
			}
		}

		// clean and close everything used for this video file
		// that is, after each video in our list of videos,
		// we need a fresh start
		onetensor_file->close();
		delete onetensor_file;

		file->close();
		delete file;
		delete gt_holder;
	}
}
