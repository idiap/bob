#include "ImageScanDataSet.h"
#include "Image.h"
#include "ipScaleYX.h"
#include "ipCrop.h"
#include "xtprobeImageFile.h"

namespace Torch {

////////////////////////////////////////////////////////////////////////////////////////////
// Constructor

ImageScanDataSet::ImageScanDataSet(	int n_files,
					const char* image_dir, char** image_files, const char* image_ext,
					const char* wnd_dir, char** wnd_files,
					int example_width, int example_height, int example_planes)
	: 	DataSet(Tensor::Short, true, Tensor::Double),
		m_n_files(n_files),
		m_image_files(new char*[m_n_files]),
		m_wnd_files(new char*[m_n_files]),
		m_n_wnds(new int[m_n_files]),
		m_cache_xtprobe(new xtprobeImageFile),
		m_cache_image(new Image(1, 1, example_planes)),
		m_cache_swimage(new Image(example_width, example_height, example_planes)),
		m_cache_fwnd(new File),
		m_cache_ifile(-1),
		m_target_pos(1),
		m_target_neg(1),
		m_ip_crop(new ipCrop),
		m_ip_scale(new ipScaleYX),
		m_rejected_bitmap(0)
{
	m_target_pos.fill(1.0);
	m_target_neg.fill(-1.0);

	// Copy the file names
	for (int i = 0; i < m_n_files; i ++)
	{
		char str[2048];
		sprintf(str, "%s/%s.%s", image_dir, image_files[i], image_ext);
		m_image_files[i] = new char[strlen(str) + 1];
		strcpy(m_image_files[i], str);

		sprintf(str, "%s/%s.wnd", wnd_dir, wnd_files[i]);
		m_wnd_files[i] = new char[strlen(str) + 1];
		strcpy(m_wnd_files[i], str);
	}

	// Check how many subwindows in each .wnd file
	m_n_examples = 0;
	for (int i = 0; i < m_n_files; i ++)
	{
		m_n_wnds[i] = 0;

		File file;
		if (file.open(m_wnd_files[i], "r") == true)
		{
			int n_samples;
			if (file.read(&n_samples, sizeof(int), 1) == 1)
			{
				m_n_examples += n_samples;
				m_n_wnds[i] = n_samples;
			}
		}
	}

	// Allocate the bit map
	const int size = m_n_examples / (8 * sizeof(unsigned int)) + 1;
	m_rejected_bitmap = new unsigned int[size];
	for (int i = 0; i < size; i ++)
	{
		m_rejected_bitmap[i] = 0x00;
	}

	// Set the parameters to the scalling operator
	m_ip_scale->setIOption("width", example_width);
	m_ip_scale->setIOption("height", example_height);
}

////////////////////////////////////////////////////////////////////////////////////////////
// Destructor

ImageScanDataSet::~ImageScanDataSet()
{
	cleanup();
}

////////////////////////////////////////////////////////////////////////////////////////////
// Delete the allocated memory

void ImageScanDataSet::cleanup()
{
	for (int i = 0; i < m_n_files; i ++)
	{
		delete[] m_image_files[i];
		delete[] m_wnd_files[i];
	}
	delete[] m_image_files;
	delete[] m_wnd_files;
	delete[] m_n_wnds;

	delete m_cache_xtprobe;
	delete m_cache_image;
	delete m_cache_swimage;
	delete m_cache_fwnd;

	delete m_ip_crop;
	delete m_ip_scale;

	delete[] m_rejected_bitmap;
}

////////////////////////////////////////////////////////////////////////////////////////////
// Access the example at the given index

Tensor* ImageScanDataSet::getExample(long index)
{
	if(!isIndex(index, m_n_examples))
		error("ImageScanDataSet::getExample - target (%d) out-of-range [0-%d].", index, m_n_examples - 1);

	// Check the image this subwindow belongs too
	int ifile = 0;
	while (	ifile < m_n_files &&
		index >= m_n_wnds[ifile])
	{
		index -= m_n_wnds[ifile];
		ifile ++;
	}

	// The required image was not loaded before or a different one was requested
	if (ifile != m_cache_ifile)
	{
		m_cache_ifile = ifile;
		if (m_cache_xtprobe->load(*m_cache_image, m_image_files[ifile]) == false)
		{
			error("ImageScanDataSet::getExample - cannot load image!");
		}

		// Load also the associated .wnd files
		m_cache_fwnd->close();
		if (m_cache_fwnd->open(m_wnd_files[ifile], "r") == false)
		{
			error("ImageScanDataSet::getExample - cannot open .wnd!");
		}
	}

	// Get the subwindow coordinates
	m_cache_fwnd->seek(sizeof(int) + index * 4 * sizeof(short), SEEK_SET);
	short sw_x, sw_y, sw_w, sw_h;
	if (	m_cache_fwnd->read(&sw_x, sizeof(short), 1) != 1 ||
		m_cache_fwnd->read(&sw_y, sizeof(short), 1) != 1 ||
		m_cache_fwnd->read(&sw_w, sizeof(short), 1) != 1 ||
		m_cache_fwnd->read(&sw_h, sizeof(short), 1) != 1)
	{
		error("ImageScanDataSet::getExample - cannot read subwindow's coordiantes!");
	}

	// Crop the image
	if (	m_ip_crop->setIOption("x", sw_x) == false ||
		m_ip_crop->setIOption("y", sw_y) == false ||
		m_ip_crop->setIOption("w", sw_w) == false ||
		m_ip_crop->setIOption("h", sw_h) == false ||
		m_ip_crop->process(*m_cache_image) == false)
	{
		error("ImageScanDataSet::getExample - failed to crop the image!");
	}

	// Scale the cropped image
	if (	m_ip_scale->process(m_ip_crop->getOutput(0)) == false)
	{
		error("ImageScanDataSet::getExample - failed to scale the cropped image!");
	}

	// Copy the result to the buffer
	m_cache_swimage->copyFrom(m_ip_scale->getOutput(0));

	return m_cache_swimage;
}

Tensor& Torch::ImageScanDataSet::operator()(long index)
{
	return *getExample(index);
}

////////////////////////////////////////////////////////////////////////////////////////////
// Access the target at the given index

Tensor* Torch::ImageScanDataSet::getTarget(long index)
{
	if(!isIndex(index, m_n_examples))
		error("ImageScanDataSet(): target (%d) out-of-range [0-%d].", index, m_n_examples - 1);

//	print("bitmap = %08.8X.%08.8X.%08.8X.%08.8X\n",
//		m_rejected_bitmap[0], m_rejected_bitmap[1], m_rejected_bitmap[2], m_rejected_bitmap[3]);

	return isRejected(index) == true ? &m_target_neg : &m_target_pos;
}

////////////////////////////////////////////////////////////////////////////////////////////
// Change the target at the given index

void Torch::ImageScanDataSet::setTarget(long index, Tensor* target)
{
	if(!isIndex(index, m_n_examples))
		error("ImageScanDataSet(): target (%d) out-of-range [0-%d].", index, m_n_examples - 1);

//	print("bitmap = %08.8X.%08.8X.%08.8X.%08.8X\n",
//		m_rejected_bitmap[0], m_rejected_bitmap[1], m_rejected_bitmap[2], m_rejected_bitmap[3]);

	if (((DoubleTensor*)target)->get(0) < 0.0)
	{
		reject(index);
	}
	else
	{
		accept(index);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////

} // namespace torch
