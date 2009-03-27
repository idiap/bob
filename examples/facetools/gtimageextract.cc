#include "FileList.h"

#include "eyecenterGTFile.h"
#include "eyecornerGTFile.h"
#include "bancaGTFile.h"
#include "cootesGTFile.h"
#include "frontalEyeNoseChinGTFile.h"
#include "halfprofileEyeNoseChinGTFile.h"
#include "profileEyeNoseChinGTFile.h"

#include "ipGeomNorm.h"

#include "Color.h"
#include "Image.h"
#include "xtprobeImageFile.h"

#include "TensorFile.h"

#include "CmdLine.h"

using namespace Torch;

///////////////////////////////////////////////////////////////////////////
// Geometric normalization parameters (check ipGeomNorm's options)
///////////////////////////////////////////////////////////////////////////

struct GeomNormParams
{
	// Constructor
	GeomNormParams()
		:	m_rotIdx1(0), m_rotIdx2(0), m_rotAngle(0.0),
			m_scaleIdx1(0), m_scaleIdx2(0), m_scaleDist(0),
			m_cropIdx1(0), m_cropIdx2(0), m_cropDx(0), m_cropDy(0),
			m_cropW(0), m_cropH(0), m_cropBorderX(0), m_cropBorderY(0)
	{
	}

	// Set the parameters as options to some ipGeomNorm
	bool		set(ipGeomNorm& gnormalizer)
	{
		CHECK_ERROR(gnormalizer.setIOption("rotIdx1", m_rotIdx1));
		CHECK_ERROR(gnormalizer.setIOption("rotIdx2", m_rotIdx2));
		CHECK_ERROR(gnormalizer.setDOption("rotAngle", m_rotAngle));
		CHECK_ERROR(gnormalizer.setIOption("scaleIdx1", m_scaleIdx1));
		CHECK_ERROR(gnormalizer.setIOption("scaleIdx2", m_scaleIdx2));
		CHECK_ERROR(gnormalizer.setIOption("scaleDist", m_scaleDist));
		CHECK_ERROR(gnormalizer.setIOption("cropIdx1", m_cropIdx1));
		CHECK_ERROR(gnormalizer.setIOption("cropIdx2", m_cropIdx2));
		CHECK_ERROR(gnormalizer.setIOption("cropDx", m_cropDx));
		CHECK_ERROR(gnormalizer.setIOption("cropDy", m_cropDy));
		CHECK_ERROR(gnormalizer.setIOption("cropW", m_cropW));
		CHECK_ERROR(gnormalizer.setIOption("cropH", m_cropH));
		CHECK_ERROR(gnormalizer.setIOption("cropBorderX", m_cropBorderX));
		CHECK_ERROR(gnormalizer.setIOption("cropBorderY", m_cropBorderY));

		return true;
	}

	///////////////////////////////////////////////////////////////////
	// Attributes

	// Rotation
	int 		m_rotIdx1, m_rotIdx2;
	double		m_rotAngle;

	// Scalling
	int		m_scaleIdx1, m_scaleIdx2;
	int		m_scaleDist;

	// Cropping
	int		m_cropIdx1, m_cropIdx2;
	int		m_cropDx, m_cropDy;
	int		m_cropW, m_cropH;
	int		m_cropBorderX, m_cropBorderY;
};

bool loadGeomNormCfg(const char* filename, GeomNormParams& params);
void saveImageGTPts(	const ShortTensor& timage, const sPoint2D* gt_pts, int n_gt_pts, const char* filename);

int main(int argc, char* argv[])
{
        ///////////////////////////////////////////////////////////////////
        // Parse the command line
        ///////////////////////////////////////////////////////////////////

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
	cmd.addSCmdOption("-image_ext", &image_ext, "pgm", "image file extension");
	cmd.addSCmdOption("-gt_ext", &gt_ext, "pos", "gt file extension");
	cmd.addICmdOption("-gt_format", &gt_format, 1, "gt format (1=eyes center, 2=banca format, 3=eyes corners, 4=eye corners + nose tip + chin, 5=left eye corners + right eye center + nose tip + chin, 6=left eye center + nose tip + chin, 7=Tim Cootes's markup 68 pts)");
	cmd.addBCmdOption("-one_gt_object", &one_gt_object, false, "if true then considers that the gt file contains one object, otherwise assumes that the first line of the file contains the number of objects");
	cmd.addBCmdOption("-verbose", &verbose, false, "verbose");
	cmd.addBCmdOption("-oimage", &oimage, false, "output intermediate images");
	cmd.addBCmdOption("-onetensor", &onetensor, false, "generates only one tensor");

	// Parse the command line
	if (cmd.read(argc, argv) < 0)
	{
		return 1;
	}

	GTFile *gt_loader = NULL;
	switch(gt_format)
	{
	case 1:
		gt_loader = new eyecenterGTFile();
		break;
	case 2:
		gt_loader = new eyecornerGTFile();
		break;
	case 3:
		gt_loader = new bancaGTFile();
		break;
	case 4:
		gt_loader = new frontalEyeNoseChinGTFile();
		break;
	case 5:
		gt_loader = new halfprofileEyeNoseChinGTFile();
		break;
	case 6:
		gt_loader = new profileEyeNoseChinGTFile();
		break;
	case 7:
		gt_loader = new cootesGTFile();
		break;
	default:
	   	warning("GT format not implemented.");
	   	return 1;
		break;
	}
	gt_loader->setBOption("verbose", verbose);

	if(one_gt_object) message("One object in GT file.");

	//
	ipGeomNorm gnormalizer;

	///////////////////////////////////////////////////////////////////
	// Parse the configuration file and set the parameters to ipGeomNorm

	GeomNormParams params;
	CHECK_FATAL(loadGeomNormCfg(cfg_norm_filename, params) == true);
	CHECK_FATAL(params.set(gnormalizer) == true);

	//
	FileList *file_list = new FileList(list_filename);


	//
	TensorFile *onetensor_file;
	const char *onetensor_filename = "onetensor.tensor";
	if(onetensor)
	{
	   	onetensor_file = new TensorFile;
		CHECK_FATAL(onetensor_file->openWrite(onetensor_filename, Tensor::Short, 2, params.m_cropH, params.m_cropW, 0, 0));

		const TensorFile::Header& onetensor_header = onetensor_file->getHeader();
		print("One tensor file:\n");
		print(" type:         [%s]\n", str_TensorTypeName[onetensor_header.m_type]);
		print(" n_dimensions: [%d]\n", onetensor_header.m_n_dimensions);
		print(" size[0]:      [%d]\n", onetensor_header.m_size[0]);
		print(" size[1]:      [%d]\n", onetensor_header.m_size[1]);
		print(" size[2]:      [%d]\n", onetensor_header.m_size[2]);
	}

	print("Number of files:%d\n", file_list->n_files);
	for(int i = 0 ; i < file_list->n_files ; i++)
	{
	   	//
		print("%s\n", file_list->file_names[i]);

		char *image_filename = new char [strlen(image_pathname) + strlen(file_list->file_names[i]) + 3 + strlen(image_ext)];
		char *gt_filename = new char [strlen(gt_pathname) + strlen(file_list->file_names[i]) + 3 + strlen(gt_ext)];

		sprintf(image_filename, "%s/%s.%s", image_pathname, file_list->file_names[i], image_ext);
		sprintf(gt_filename, "%s/%s.%s", gt_pathname, file_list->file_names[i], gt_ext);

		//
		print("Image file: %s\n", image_filename);

		Image image(1, 1, 3);
		xtprobeImageFile xtprobe;
		CHECK_FATAL(xtprobe.load(image, image_filename) == true);

		print("   width   = %d\n", image.size(1));
		print("   height  = %d\n", image.size(0));
		print("   nplanes = %d\n", image.size(2));

		//
		print("GT file: %s\n", gt_filename);

		File gt_file;
		gt_file.open(gt_filename, "r");
		if(one_gt_object) gt_loader->load(&gt_file);
		else
		{
		   	int n;
		   	gt_file.scanf("%d", &n);
			print("Number of objects: %d\n", n);
			for(int j = 0 ; j < n ; j++)
				gt_loader->load(&gt_file);
		}
		gt_file.close();


		///////////////////////////////////////////////////////////////////
		// Geometric normalize the image and save the result

		CHECK_FATAL(gnormalizer.setGTPoints(gt_loader->getPoints(), gt_loader->getNPoints()) == true);
		CHECK_FATAL(gnormalizer.process(image) == true);

		const ShortTensor& norm_timage = (const ShortTensor&)gnormalizer.getOutput(0);

		print("Output image size:\n");
		print("   width   = %d\n", norm_timage.size(1));
		print("   height  = %d\n", norm_timage.size(0));

		// Convert the output color image (3D tensor RGB) to a grayscale image (3D gray)
		Image imagegray(norm_timage.size(1), norm_timage.size(0), 1);
		imagegray.copyFrom((Image &)norm_timage); // the cast is necessary other copyFrom will not consider it as an image and will not convert it to grayscal
        
		//
		if(onetensor)
		{
			// Select the grayscale channel as a 2D tensor and save it !
			ShortTensor *t_ = new ShortTensor();
			t_->select(&imagegray, 2, 0);
			onetensor_file->save(*t_);
			delete t_;
		}
		else
		{
			TensorFile tensor_file;

			print("Writing tensor file ...\n");
			char* tensor_filename = new char [strlen(file_list->file_names[i]) + 8];
			sprintf(tensor_filename, "%s.tensor", file_list->file_names[i]);
			CHECK_FATAL(tensor_file.openWrite(tensor_filename, Tensor::Short, 2, norm_timage.size(0), norm_timage.size(1), 0, 0));

			const TensorFile::Header& header = tensor_file.getHeader();
			print("Tensor file:\n");
			print(" type:         [%s]\n", str_TensorTypeName[header.m_type]);
			print(" n_dimensions: [%d]\n", header.m_n_dimensions);
			print(" size[0]:      [%d]\n", header.m_size[0]);
			print(" size[1]:      [%d]\n", header.m_size[1]);
			print(" size[2]:      [%d]\n", header.m_size[2]);
			//print(" size[3]:      [%d]\n", header.m_size[3]);

			// Select the grayscale channel as a 2D tensor and save it !
			ShortTensor *t_ = new ShortTensor();
			t_->select(&imagegray, 2, 0);
			tensor_file.save(*t_);
			delete t_;

			tensor_file.close();
			delete [] tensor_filename;
		}

		if(oimage)
		{
			char *output_image_filename = new char [strlen(file_list->file_names[i]) + 15];
			sprintf(output_image_filename, "%s.original.jpeg", file_list->file_names[i]);
			saveImageGTPts(image, gt_loader->getPoints(), gt_loader->getNPoints(), output_image_filename);
			sprintf(output_image_filename, "%s.gtnorm.jpeg", file_list->file_names[i]);
			saveImageGTPts(norm_timage, gnormalizer.getNMPoints(), gt_loader->getNPoints(), output_image_filename);
			sprintf(output_image_filename, "%s.geomnorm.jpeg", file_list->file_names[i]);
			saveImageGTPts(norm_timage, 0, 0, output_image_filename);
			delete [] output_image_filename;
		}

		//
		delete [] gt_filename;
		delete [] image_filename;
	}

	if(onetensor)
	{
		onetensor_file->close();
		delete onetensor_file;
	}

	delete file_list;
	delete gt_loader;

        // OK
	return 0;
}

///////////////////////////////////////////////////////////////////////////
// Parse a geometrical normalization configuration file
///////////////////////////////////////////////////////////////////////////

const char* normString(const char* str)
{
	const char* ret = str;
	while (*ret != '\0' && *ret != ':')
	{
		ret ++;
	}
	if (*ret == ':')
	{
		ret ++;
	}
	return ret;
}

bool loadGeomNormCfg(const char* filename, GeomNormParams& params)
{
	// Open the file
	File file;
	if (file.open(filename, "r") == false)
	{
		return false;
	}

	const int sizeBuf = 512;
	char str[sizeBuf];

	// Read the rotation parameters
	if (	file.gets(str, sizeBuf) == 0 ||
		sscanf(normString(str), "%d %d %lf",
			&params.m_rotIdx1, &params.m_rotIdx2, &params.m_rotAngle) != 3)
	{
		print("[%s] - [%s]\n", str, normString(str));
		return false;
	}

	// Read the scalling parameters
	if (	file.gets(str, sizeBuf) == 0 ||
		sscanf(normString(str), "%d %d %d",
			&params.m_scaleIdx1, &params.m_scaleIdx2, &params.m_scaleDist) != 3)
	{
		print("[%s] - [%s]\n", str, normString(str));
		return false;
	}

	// Read the cropping parameters
	if (	file.gets(str, sizeBuf) == 0 ||
		sscanf(normString(str), "%d %d %d %d %d %d %d %d",
			&params.m_cropIdx1, &params.m_cropIdx2,
			&params.m_cropDx, &params.m_cropDy,
			&params.m_cropW, &params.m_cropH,
			&params.m_cropBorderX, &params.m_cropBorderY) != 8)
	{
		print("[%s] - [%s]\n", str, normString(str));
		return false;
	}

	file.close();

	// Verbose
	print("\nRotation: pt1 = %d, pt2 = %d, angle = %lf\n",
		params.m_rotIdx1, params.m_rotIdx2, params.m_rotAngle);
	print("Scalling: pt1 = %d, pt2 = %d, distance = %d\n",
		params.m_scaleIdx1, params.m_scaleIdx2, params.m_scaleDist);
	print("Cropping: pt1 = %d, pt2 = %d, dx = %d, dy = %d, w = %d, h = %d, borderx = %d, bordery = %d\n",
		params.m_cropIdx1, params.m_cropIdx2, params.m_cropDx, params.m_cropDy,
		params.m_cropW, params.m_cropH, params.m_cropBorderX, params.m_cropBorderY);

	// OK
	return true;
}

///////////////////////////////////////////////////////////////////////////
// Save an image and draw the ground truth on top
///////////////////////////////////////////////////////////////////////////

void saveImageGTPts(	const ShortTensor& timage, const sPoint2D* gt_pts, int n_gt_pts, const char* filename)
{
	const int h = timage.size(0);
	const int w = timage.size(1);

	Image image(w, h, timage.size(2));
	image.copyFrom(timage);
	for (int i = 0; i < n_gt_pts; i ++)
	{
		const int x = gt_pts[i].x;
		const int y = gt_pts[i].y;
		if (	x > 4 && x + 4 < w &&
			y > 4 && y + 4 < h)
		{
			image.drawLine(x - 4, y, x + 4, y, Torch::red);
			image.drawLine(x, y - 4, x, y + 4, Torch::red);
		}
	}

	xtprobeImageFile xtprobe;
	xtprobe.save(image, filename);
}

