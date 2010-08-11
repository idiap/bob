#ifndef _IMAGE_SCAN_DATA_SET_INC
#define _IMAGE_SCAN_DATA_SET_INC

#include "core/DataSet.h"

namespace Torch
{
	class Image;
	class xtprobeImageFile;
	class ipScaleYX;
	class ipCrop;
	class File;

	//////////////////////////////////////////////////////////////////////////////////////
	// Torch::ImageScanDataSet:
	//	Designed to deal with large number of (negative) sample subwindows.
	//
	//	Generates samples from a list of .wnd files and associated images.
	//	It uses an Image of fixed size as a buffered example, requested with the index.
	//	By default, each subwindow has the target <1>; to reject some subwindow
	//		set its target as <-1>.
	//
	// TODO: doxygen header!
	//////////////////////////////////////////////////////////////////////////////////////

	class ImageScanDataSet : public DataSet
	{
	public:

		// Constructor
		ImageScanDataSet(	int n_files,
					const char* image_dir, char** image_files, const char* image_ext,
					const char* wnd_dir, char** wnd_files,
					int example_width, int example_height, int example_planes);

		// Destructor
		virtual ~ImageScanDataSet();

		// Do _not_ try to modify them, they are readonly!
		virtual Tensor* getExample(long index);
		virtual Tensor&	operator()(long index);

		// Access targets
		virtual Tensor* getTarget(long index);
		virtual void	setTarget(long index, Tensor* target);

		// Load a tensor data file
		bool		load(const char* file_name);

	private:

		//////////////////////////////////////////////////////

		// Delete the allocated memory
		void 		cleanup();

		// Test if some target is rejected or not
		bool		isRejected(long index) const
		{
			static const unsigned int ui_size = 8 * sizeof(unsigned int);

			return (m_rejected_bitmap[index / ui_size] & (((unsigned int)0x01) << index % ui_size)) != 0x00;
		}

		// Reject some target
		void		reject(long index) const
		{
			static const unsigned int ui_size = 8 * sizeof(unsigned int);

			m_rejected_bitmap[index / ui_size] |= (((unsigned int)0x01) << index % ui_size);
		}

		// Accept some target
		void		accept(long index) const
		{
			static const unsigned int ui_size = 8 * sizeof(unsigned int);

			m_rejected_bitmap[index / ui_size] &= ~(((unsigned int)0x01) << index % ui_size);
		}

		//////////////////////////////////////////////////////
		// Attributes

		// <image, subwindows> correspondence
		int		m_n_files;
		char**		m_image_files;	// Image files
		char**		m_wnd_files;	// Subwindow coordinates files
		int*		m_n_wnds;	// Number of subwindows in each .wnd file

		// Caching
		xtprobeImageFile*	m_cache_xtprobe;
		Image*			m_cache_image;
		Image*			m_cache_swimage;
		File*			m_cache_fwnd;
		int			m_cache_ifile;

		// Buffered target tensor - rejected / not rejected
		DoubleTensor	m_target_pos;	// +1 - not rejected
		DoubleTensor	m_target_neg;	// -1 - rejected

		// Object to crop and scale to the required size
		ipCrop*		m_ip_crop;
		ipScaleYX*	m_ip_scale;

		// Keep track of each sample subwindow if it is rejected or not (as bits)
		unsigned int*	m_rejected_bitmap;
	};
}

#endif

