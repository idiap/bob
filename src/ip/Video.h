#ifndef VIDEO_INC
#define VIDEO_INC

#include "core/Object.h"

namespace Torch
{
	class Image;

	// Hides the implementation details
	namespace Private
	{
		class VideoImpl;
	}

	/////////////////////////////////////////////////////////////////////////
	/** This class is designed to handle video reading and writing.

	    @author Sebastien Marcel (marcel@idiap.ch)
	    @version 2.0
	    \date
	    @since 1.0
	*/
	//	- PARAMETERS (name, type, default value, description):
	//		"width"		int	-1	"video frame width"
	//		"height"	int	-1	"video frame height"
	//		"bitrate"	float	1500000	"bitrate"
	//		"framerate"	float	25	"framerate"
	//		"gop"		int	100	"gop size"
	/////////////////////////////////////////////////////////////////////////
	class Video : public Object
	{
	public:

		// Video processing stages
		enum State
		{
			Idle,
			Read,		// Loading some video, moving through frames
			Write		// Saving some video, adding frames one by one
		};

		/// Constructor
		Video(const char* filename = 0, const char *open_flags = "r");

		/// Destructor
		~Video();

		/// Open a video file for reading or writting
		bool			open(const char* filename, const char *open_flags = "r");

		/// Close the video file (if opened)
		void			close();

		/// Read the next frame in the video file (if any)
		bool			read(Image& image);

		/// Write the frame to the video file
		bool			write(const Image& image);

		/// Returns the name of the codec
		const char*		codec() const;

		/////////////////////////////////////////////////////////////////////////////////////////
		/// Access functions

		int			getNFrames() const;
		State			getState() const;

		// NB: Use getXOption to retrieve width, height, bitrate, framerate

	protected:

		/// called when some option was changed - overriden
		virtual void		optionChanged(const char* name);

	private:

		/////////////////////////////////////////////////////////////////////////////////////
		/// Attributes

		Private::VideoImpl*	m_impl;		// Implementation details
		unsigned char*		m_pixmap;
	};
}

#endif
