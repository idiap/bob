#include "Video.h"
#include "Image.h"

#ifdef HAVE_FFMPEG
extern "C"
{
	#include "ffmpeg/avformat.h"
	// WARNING: if you use libswscale below the code becomes GPL !!!
	#include "ffmpeg/swscale.h"

	// Example input program: http://web.me.com/dhoerl/Home/Tech_Blog/Entries/2009/1/22_Revised_avcodec_sample.c.html
	// Example output program: http://cekirdek.pardus.org.tr/~ismail/ffmpeg-docs/output-example_8c-source.html
}
#endif

namespace Torch {

	namespace Private
	{
		//////////////////////////////////////////////////////////////////////////////////////////////////
		// Singleton object used for initializing only ONCE the ffmpeg library
		//////////////////////////////////////////////////////////////////////////////////////////////////
	#ifdef HAVE_FFMPEG

		struct ffmpeg_init
		{
		private:
			// Constructor
			ffmpeg_init()
			{
				avcodec_init();
				//avcodec_register_all();
				av_register_all();

				/* Display FFmpeg warning/errors
					-1: show nothing
					0 : show warning
					1 : show warning + errors
					2 : debug mode
				*/
				av_log_set_level(-1);
			}

		public:

			// Destructor
			~ffmpeg_init()
			{
			}

			// Access to the only instance
			static const ffmpeg_init& getInstance()
			{
				static const ffmpeg_init the_instance;
				return the_instance;
			}

			// Dummy function to remove some warnings
			void do_nothing() const
			{
				//
			}

			// Don't allow copying
			ffmpeg_init(const ffmpeg_init& other);
			ffmpeg_init& operator=(const ffmpeg_init& other);
		};

		//////////////////////////////////////////////////////////////////////////////////////////////////
		// Specific ffmpeg structures for reading video files
		//////////////////////////////////////////////////////////////////////////////////////////////////
		struct ffmpeg_read
		{
			// Constructor
			ffmpeg_read()
			{
				init();
			}

			// Destructor
			~ffmpeg_read()
			{
				cleanup();
			}

			// Initialize
			void init()
			{
				pFormatCtx = 0;
				i = videoStream = 0;
				pCodecCtx = 0;
				pCodec = 0;
				pFrame = 0;
				pFrameRGB = 0;
				frameFinished = 0;
				numBytes = 0;
				buffer = 0;
			}

			// Cleanup
			void cleanup()
			{
				// Free the RGB image
				if (buffer != 0)
				{
					free(buffer);
				}
				if (pFrameRGB != 0)
				{
					av_free(pFrameRGB);
				}

				// Free the YUV frame
				if (pFrame != 0)
				{
					av_free(pFrame);
				}

				// Close the codec
				if (pCodecCtx != 0)
				{
					avcodec_close(pCodecCtx);
				}

				// Close the video file
				if (pFormatCtx != 0)
				{
					av_close_input_file(pFormatCtx);
				}

				init();
			}

			// Attributes
			AVFormatContext *pFormatCtx;
			int             i, videoStream;
			AVCodecContext  *pCodecCtx;
			AVCodec         *pCodec;
			AVFrame         *pFrame;
			AVFrame         *pFrameRGB;
			AVPacket        packet;
			int             frameFinished;
			int             numBytes;
			uint8_t         *buffer;
		};

		//////////////////////////////////////////////////////////////////////////////////////////////////
		// Specific ffmpeg structures for writing video files
		//////////////////////////////////////////////////////////////////////////////////////////////////
		struct ffmpeg_write
		{
			// Constructor
			ffmpeg_write()
			{
				init();
			}

			// Destructor
			~ffmpeg_write()
			{
				cleanup();
			}

			// Initialize
			void init()
			{
				fmt = 0;
				oc = 0;
				video_st = 0;
				video_pts = 0.0;
				i = 0;
				picture = 0;
				tmp_picture = 0;
				video_outbuf = 0;
				frame_count = 0;
				video_outbuf_size = 0;
			}

			// Cleanup
			void cleanup()
			{
				// write the trailer, if any.  the trailer must be written
				// before you close the CodecContexts open when you wrote the
				// header; otherwise write_trailer may try to use memory that
				// was freed on av_codec_close()
				if (oc != 0)
				{
					av_write_trailer(oc);
				}

				// close each codec
				if (video_st != 0)
				{
					close_video();
				}

				// free the streams
				if (oc != 0)
				{
					for(i = 0; i < oc->nb_streams; i++)
					{
						av_freep(&oc->streams[i]->codec);
						av_freep(&oc->streams[i]);
					}
				}

				// close the output file
				if (fmt != 0 && !(fmt->flags & AVFMT_NOFILE))
				{
					url_fclose(&oc->pb);
				}

				// free the stream
				if (oc != 0)
				{
					av_free(oc);
				}

				init();
			}

			// Add a video output stream
			AVStream* add_video_stream(int width, int height, float bitrate, float framerate, int gop)
			{
				AVCodecContext *c;
				AVStream *st;

				st = av_new_stream(oc, 0);
				if (!st)
				{
					warning("Video::add_video_stream - could not alloc stream\n");
					return NULL;
				}

				c = st->codec;
				c->codec_id = fmt->video_codec;
				c->codec_type = CODEC_TYPE_VIDEO;

				// put sample parameters
				c->bit_rate = bitrate;
				// resolution must be a multiple of two
				c->width = width;
				c->height = height;
				// time base: this is the fundamental unit of time (in seconds) in terms
				//      of which frame timestamps are represented. for fixed-fps content,
				//	timebase should be 1/framerate and timestamp increments should be
				//	identically 1.
				c->time_base.den = framerate;
				c->time_base.num = 1;
				c->gop_size = gop; // emit one intra frame every N frames at most
				c->pix_fmt = PIX_FMT_YUV420P;
				if (c->codec_id == CODEC_ID_MPEG2VIDEO)
				{
					// just for testing, we also add B frames
					c->max_b_frames = 2;
				}
				if (c->codec_id == CODEC_ID_MPEG1VIDEO)
				{
					// Needed to avoid using macroblocks in which some coeffs overflow.
					// This does not happen with normal video, it just happens here as
					// the motion of the chroma plane does not match the luma plane.
					c->mb_decision=2;
				}
				// some formats want stream headers to be separate
				if (oc->oformat->flags & AVFMT_GLOBALHEADER)
					c->flags |= CODEC_FLAG_GLOBAL_HEADER;

				return st;
			}

			// Allocate a buffer picture
			AVFrame* alloc_picture(enum PixelFormat pix_fmt, int width, int height)
			{
				AVFrame *picture;
				uint8_t *picture_buf;
				int size;

				picture = avcodec_alloc_frame();
				if (!picture)
					return NULL;
				size = avpicture_get_size(pix_fmt, width, height);
				picture_buf = (uint8_t*)av_malloc(size);
				if (!picture_buf)
				{
					av_free(picture);
					return NULL;
				}
				avpicture_fill((AVPicture *)picture, picture_buf,
						pix_fmt, width, height);
				return picture;
			}

			// Open the video stream
			bool open_video()
			{
				AVCodec *codec;
				AVCodecContext *c;

				c = video_st->codec;

				// find the video encoder
				codec = avcodec_find_encoder(c->codec_id);
				if (!codec)
				{
					warning("Video::open_video - codec not found\n");
					return false;
				}

				// open the codec
				if (avcodec_open(c, codec) < 0)
				{
					warning("Video::open_video - could not open codec\n");
					return false;
				}

				video_outbuf = NULL;
				if (!(oc->oformat->flags & AVFMT_RAWPICTURE))
				{
					// allocate output buffer
					// XXX: API change will be done
					//	buffers passed into lav* can be allocated any way you prefer,
					//	as long as they're aligned enough for the architecture, and
					//	they're freed appropriately (such as using av_free for buffers
					//	allocated with av_malloc)
					video_outbuf_size = 200000;
					video_outbuf = (uint8_t*)av_malloc(video_outbuf_size);
				}

				// allocate the encoded raw picture
				picture = alloc_picture(c->pix_fmt, c->width, c->height);
				if (!picture)
				{
					warning("Video::open_video - could not allocate picture\n");
					return false;
				}

				// if the output format is not RGB24, then a temporary RGB24
				//	picture is needed too. It is then converted to the required
				// 	output format
				tmp_picture = NULL;
				if (c->pix_fmt != PIX_FMT_RGB24)
				{
					tmp_picture = alloc_picture(PIX_FMT_RGB24, c->width, c->height);
					if (!tmp_picture)
					{
						warning("Video::open_video - could not allocate temporary picture\n");
						return false;
					}
				}

				// OK
				return true;
			}

			// Write a video frame
			bool write_video_frame(unsigned char* pixmap)
			{
				int out_size, ret;
				AVCodecContext *c;
				static struct SwsContext *img_convert_ctx = NULL;

				c = video_st->codec;

				if (false) //frame_count >= STREAM_NB_FRAMES)
				{
					// no more frame to compress. The codec has a latency of a few
					//     	frames if using B frames, so we get the last frames by
					//	passing the same picture again
				}
				else
				{
					if (c->pix_fmt != PIX_FMT_RGB24)
					{
						// as we only generate a RGB24 picture, we must convert it
						//	to the codec pixel format if needed
						if (img_convert_ctx == NULL)
						{
							img_convert_ctx = sws_getContext(c->width, c->height,
											PIX_FMT_RGB24,
											c->width, c->height,
											c->pix_fmt,
											SWS_BICUBIC, NULL, NULL, NULL);
							if (img_convert_ctx == NULL)
							{
								warning("Video::write_video_frame - cannot initialize the conversion context\n");
								return false;
							}
						}

						// replace data in the buffer frame by the pixmap to encode
						tmp_picture->linesize[0] = c->width*3;
						memcpy(tmp_picture->data[0], pixmap, 3*c->width*c->height);
						sws_scale(img_convert_ctx, tmp_picture->data, tmp_picture->linesize,
							0, c->height, picture->data, picture->linesize);
					}
					else
					{
						picture->linesize[0] = c->width*3;
						memcpy(picture->data[0], pixmap, 3*c->width*c->height);
					}
				}

				if (oc->oformat->flags & AVFMT_RAWPICTURE)
				{
					// raw video case. The API will change slightly in the near future for that
					AVPacket pkt;
					av_init_packet(&pkt);

					pkt.flags |= PKT_FLAG_KEY;
					pkt.stream_index= video_st->index;
					pkt.data= (uint8_t *)picture;
					pkt.size= sizeof(AVPicture);

					ret = av_interleaved_write_frame(oc, &pkt);
				}
				else
				{
					// encode the image
					out_size = avcodec_encode_video(c, video_outbuf, video_outbuf_size, picture);
					// if zero size, it means the image was buffered
					if (out_size > 0)
					{
						AVPacket pkt;
						av_init_packet(&pkt);

						if (c->coded_frame->pts != AV_NOPTS_VALUE)
							pkt.pts= av_rescale_q(c->coded_frame->pts, c->time_base, video_st->time_base);
						if(c->coded_frame->key_frame)
							pkt.flags |= PKT_FLAG_KEY;
						pkt.stream_index= video_st->index;
						pkt.data= video_outbuf;
						pkt.size= out_size;

						// write the compressed frame in the media file
						ret = av_interleaved_write_frame(oc, &pkt);
					}
					else
					{
						ret = 0;
					}
				}
				if (ret != 0)
				{
					warning("Video::write_video_frame - error while writing video frame\n");
					return false;
				}

				// OK
				frame_count ++;
				return true;
			}

			// Close the video codec
			void close_video()
			{
				avcodec_close(video_st->codec);
				av_free(picture->data[0]);
				av_free(picture);
				if (tmp_picture)
				{
					av_free(tmp_picture->data[0]);
					av_free(tmp_picture);
				}
				av_free(video_outbuf);
			}

			// Attributes
			AVOutputFormat *fmt;
			AVFormatContext *oc;
			AVStream *video_st;
			double video_pts;
			int i;
			AVFrame *picture, *tmp_picture;
			uint8_t *video_outbuf;
			int frame_count, video_outbuf_size;
		};
	#endif

		//////////////////////////////////////////////////////////////////////////////////////////////////
		// VideoImpl - loading/saving video implementation
		/////////////////////////////////////////////////////////////////////////////////////////////////
		class VideoImpl
		{
		public:
			// Constructor
			VideoImpl(bool verbose, const char* filename = 0, const char *open_flags = "r")
				:	m_state(Video::Idle),
					m_nframes(0),
					m_width(0), m_height(0),
					m_bitrate(1500000.0f), m_framerate(25.0f), m_gop(100)
			{
			#ifndef HAVE_FFMPEG
				error("HAVE_FFMPEG not defined in Video::Video");
			#endif

			#ifdef HAVE_FFMPEG
				if (verbose)
				{
					print("LIBAVCODEC_VERSION = %d (%s)\n", LIBAVCODEC_VERSION_INT, AV_STRINGIFY(LIBAVCODEC_VERSION));
					print("LIBAVFORMAT_VERSION = %d (%s)\n", LIBAVFORMAT_VERSION_INT, AV_STRINGIFY(LIBAVFORMAT_VERSION));
				}

				// Initialize the ffmpeg library
				static const ffmpeg_init& ffmpeg_init_instance = ffmpeg_init::getInstance();
				ffmpeg_init_instance.do_nothing();

				// Open the video (if any given)
				if (filename != 0)
				{
					open(filename, open_flags);
				}
			#endif
			}

			// Destructor
			~VideoImpl()
			{
				close();
			}

			// Open a video file for reading or writting
			bool open(bool verbose, const char* filename, const char *open_flags = "r")
			{
			#ifdef HAVE_FFMPEG
				if (strcmp(open_flags, "r") == 0)
				{
					// Open video file
					if (av_open_input_file(&m_fread.pFormatCtx, filename, NULL, 0, NULL) != 0)
					{
						warning("Video::open - impossible to open video file %s.", filename);
						cleanup();
						return false;
					}

					// Retrieve stream information
					if (av_find_stream_info(m_fread.pFormatCtx) < 0)
					{
						warning("Video::open - impossible to find AV streams in the file.");
						cleanup();
						return false;
					}

					// Dump information about file onto standard error
					dump_format(m_fread.pFormatCtx, 0, filename, false);

					// Find the first video stream
					m_fread.videoStream = -1;
					for (m_fread.i = 0; m_fread.i < m_fread.pFormatCtx->nb_streams; m_fread.i ++)
						if (m_fread.pFormatCtx->streams[m_fread.i]->codec->codec_type
							== CODEC_TYPE_VIDEO)
						{
							m_fread.videoStream = m_fread.i;
							break;
						}
					if (m_fread.videoStream == -1)
					{
						warning("Video::open - impossible to find a video stream.");
						cleanup();
						return false;
					}

					// Get a pointer to the codec context for the video stream
					m_fread.pCodecCtx = m_fread.pFormatCtx->streams[m_fread.videoStream]->codec;

					// Find the decoder for the video stream
					m_fread.pCodec = avcodec_find_decoder(m_fread.pCodecCtx->codec_id);
					if (m_fread.pCodec == NULL)
					{
						warning("Video::open - unknown video codec.");
						cleanup();
						return false;
					}

					// Open codec
					if (avcodec_open(m_fread.pCodecCtx, m_fread.pCodec) < 0)
					{
						warning("Video::open - impossible to open the codec.");
						cleanup();
						return false;
					}

					// Hack to correct wrong frame rates that seem to be generated by some codecs
					if(m_fread.pCodecCtx->time_base.num > 1000 &&
						m_fread.pCodecCtx->time_base.den == 1)
							m_fread.pCodecCtx->time_base.den = 1000;

					// Allocate video frames
					m_fread.pFrame = avcodec_alloc_frame();
					m_fread.pFrameRGB = avcodec_alloc_frame();
					if (m_fread.pFrame == NULL || m_fread.pFrameRGB == NULL)
					{
						warning("Video::open - impossible to alloc the decoded frames.");
						cleanup();
						return false;
					}

					// Determine required buffer size and allocate buffer
					m_fread.numBytes = avpicture_get_size(	PIX_FMT_RGB24,
									m_fread.pCodecCtx->width,
									m_fread.pCodecCtx->height);
					m_fread.buffer = (uint8_t*)malloc(m_fread.numBytes);

					// Assign appropriate parts of buffer to image planes in pFrameRGB
					avpicture_fill((AVPicture *)m_fread.pFrameRGB, m_fread.buffer, PIX_FMT_RGB24,
							m_fread.pCodecCtx->width, m_fread.pCodecCtx->height);

					m_fread.i = 0;

					// Save information about the video
					if (m_fread.pFormatCtx->streams[m_fread.videoStream]->nb_frames > 0)
					{
						// The number of frames are known
						m_nframes = m_fread.pFormatCtx->streams[m_fread.videoStream]->nb_frames;
						m_framerate = m_nframes * AV_TIME_BASE / m_fread.pFormatCtx->duration;
					}
					else
					{
						// The number of frames is not known
						m_framerate = av_q2d(m_fread.pFormatCtx->streams[m_fread.videoStream]->r_frame_rate);
						m_nframes = (int)(m_framerate * m_fread.pFormatCtx->duration / AV_TIME_BASE);
					}
					m_width = m_fread.pCodecCtx->width;
					m_height = m_fread.pCodecCtx->height;
					m_bitrate = m_fread.pFormatCtx->bit_rate;

					// OK
					m_state = Video::Read;
					return true;
				}

				else if (strcmp(open_flags, "w") == 0)
				{
					// check parameters
					if (m_width == -1 || m_height == -1)
					{
						warning("Video::open - Please set all options (width, height, fps, bitrate, ...) before doing open().");
						return false;
					}

					// auto detect the output format from the name. default is mpeg.
					m_fwrite.fmt = guess_format(NULL, filename, NULL);
					if (!m_fwrite.fmt)
					{
						warning("Video::open - could not deduce output format from file extension: using MPEG.");
						m_fwrite.fmt = guess_format("mpeg", NULL, NULL);
					}
					if (!m_fwrite.fmt)
					{
						warning("Video::open - could not find suitable output format.");
						cleanup();
						return false;
					}

					if(verbose) print("Video::open - opening video file %s in write mode.\n", filename);

					// allocate the output media context
					m_fwrite.oc = av_alloc_format_context();
					if (!m_fwrite.oc)
					{
						warning("Video::open - impossible to alloc an output format context");
						cleanup();
						return false;
					}
					m_fwrite.oc->oformat = m_fwrite.fmt;
					snprintf(m_fwrite.oc->filename, sizeof(m_fwrite.oc->filename), "%s", filename);
					snprintf(m_fwrite.oc->title, sizeof(m_fwrite.oc->title), "%s", filename);
					snprintf(m_fwrite.oc->author, sizeof(m_fwrite.oc->author), "%s", "Sébastien Marcel (marcel@idiap.ch) -- Idiap Research Institute (www.idiap.ch)");
					snprintf(m_fwrite.oc->copyright, sizeof(m_fwrite.oc->copyright), "%s", "(c) Torch5spro 2010");
					snprintf(m_fwrite.oc->comment, sizeof(m_fwrite.oc->comment), "%s", "Created by Torch5spro using FFmpeg");
					snprintf(m_fwrite.oc->album, sizeof(m_fwrite.oc->album), "%s", "Torch5spro");
					snprintf(m_fwrite.oc->genre, sizeof(m_fwrite.oc->genre), "%s", "Research");
					m_fwrite.oc->year = 2010;
					m_fwrite.oc->track = 0;

					// add the video stream using the default format codecs and initialize the codec
					m_fwrite.video_st = NULL;
					if (m_fwrite.fmt->video_codec != CODEC_ID_NONE)
					{
						m_fwrite.video_st = m_fwrite.add_video_stream(
									m_width, m_height, m_bitrate, m_framerate, m_gop);
					}
					if (m_fwrite.video_st == NULL)
					{
						warning("Video::open - impossible to allocate a new video stream.");
						cleanup();
						return false;
					}

					// set the output parameters (must be done even if no parameters).
					if (av_set_parameters(m_fwrite.oc, NULL) < 0)
					{
						warning("Video::open - invalid output format parameters.");
						cleanup();
						return false;
					}

					dump_format(m_fwrite.oc, 0, filename, 1);

					// now that all the parameters are set, we can open the video codecs
					//	and allocate the necessary encode buffers
					if (m_fwrite.open_video() == false)
					{
						warning("Video::open - cannot open the video codecs.");
						cleanup();
						return false;
					}

					// open the output file, if needed
					if (!(m_fwrite.fmt->flags & AVFMT_NOFILE))
					{
						if (url_fopen(&m_fwrite.oc->pb, filename, URL_WRONLY) < 0)
						{
							warning("Video::open - impossible to open the file '%s'.", filename);
							cleanup();
							return false;
						}
					}

					// write the stream header, if any
					av_write_header(m_fwrite.oc);

					// OK
					m_state = Video::Write;
					return true;
				}

				return false;
			#endif

				warning("VideoFile::open - ffmpeg not supported.");
				return false;
			}

			// Close the video file (if opened)
			void close()
			{
			#ifdef HAVE_FFMPEG
				cleanup();
				return;
			#endif

				warning("Video::close - ffmpeg not supported.");
			}

			// Read the next frame in the video file (if any)
			bool read(bool verbose)
			{
				if (m_state != Video::Read)
				{
					warning("Video::read - impossible to read, the video is not open in read mode.\n");
					return false;
				}

			#ifdef HAVE_FFMPEG

				bool ok = false;
				while (av_read_frame(m_fread.pFormatCtx, &m_fread.packet) >= 0)
				{
					// Is this a packet from the video stream?
					if (m_fread.packet.stream_index == m_fread.videoStream)
					{
						// Decode video frame
						avcodec_decode_video(m_fread.pCodecCtx, m_fread.pFrame, &m_fread.frameFinished,
								m_fread.packet.data, m_fread.packet.size);

						// Did we get a video frame?
						if (m_fread.frameFinished)
						{
							static struct SwsContext *img_convert_ctx = 0;

							// Convert the image into YUV format that SDL uses
							if (img_convert_ctx == NULL)
							{
								const int w = m_fread.pCodecCtx->width;
								const int h = m_fread.pCodecCtx->height;

								img_convert_ctx = sws_getContext(w, h,
												m_fread.pCodecCtx->pix_fmt,
												w, h, PIX_FMT_RGB24, SWS_BICUBIC,
												NULL, NULL, NULL);
								if (img_convert_ctx == NULL)
								{
									warning("Video::read - cannot initialize the conversion context!\n");
									av_free_packet(&m_fread.packet);
									return false;
								}
							}
							int ret = sws_scale(	img_convert_ctx,
										m_fread.pFrame->data, m_fread.pFrame->linesize,
										0, m_fread.pCodecCtx->height,
										m_fread.pFrameRGB->data, m_fread.pFrameRGB->linesize);

							// Got the image - exit
							m_fread.i ++;
							ok = true;

							// Free the packet that was allocated by av_read_frame
							av_free_packet(&m_fread.packet);
							break;
						}
					}

					// Free the packet that was allocated by av_read_frame
					av_free_packet(&m_fread.packet);
				}

				return ok;

			#endif

				warning("Video::read - ffmpeg not supported.");
				return false;
			}

			// Write the frame to the video file
			bool write(bool verbose, unsigned char* pixmap)
			{
				if (m_state != Video::Write)
				{
					warning("Video::write - impossible to write, the video is not open in write mode.\n");
					return false;
				}

			#ifdef HAVE_FFMPEG
				return m_fwrite.write_video_frame(pixmap);
			#endif

				warning("Video::write - ffmpeg not supported.");
				return false;
			}

			// Returns the name of the codec
			const char* codec() const
			{
			#ifdef HAVE_FFMPEG
				return m_state == Video::Read ? m_fread.pCodec->name : 0;
			#endif
				warning("Video::codec - ffmpeg not supported.");
				return 0;
			}

			// Change the video parameters
			void setup(int width, int height, float bitrate, float framerate, int gop)
			{
				m_width = width;
				m_height = height;
				m_bitrate = bitrate;
				m_framerate = framerate;
				m_gop = gop;
			}

			/////////////////////////////////////////////////////////////////////////////////////////
			/// Access functions

			int			getNFrames() const { return m_nframes; }
			Video::State		getState() const { return m_state; }
			int			getWidth() const { return m_width; }
			int			getHeight() const { return m_height; }
			float			getBitrate() const { return m_bitrate; }
			float			getFramerate() const { return m_framerate; }
			int			getGop() const { return m_gop; }
			const unsigned char*	getPixmap() const
			{
			#ifdef HAVE_FFMPEG
				return m_state == Video::Read ? m_fread.pFrameRGB->data[0] : 0;
			#endif
				warning("Video::getPixmap - ffmpeg not supported.");
				return 0;
			}

			////////////////////////////////////////////////////////////////////////////

		private:

			// Deallocate the ffmpeg structures
			void cleanup()
			{
			#ifdef HAVE_FFMPEG
				m_fwrite.cleanup();
				m_fread.cleanup();
			#endif
				m_state = Video::Idle;
			}

			////////////////////////////////////////////////////////////////////////////
			// Attributes

		#ifdef HAVE_FFMPEG
			ffmpeg_read		m_fread;
			ffmpeg_write		m_fwrite;
		#endif

			// Video statistics
			Video::State 		m_state;
			int 			m_nframes, m_width, m_height, m_gop;
			float 			m_bitrate, m_framerate;
		};
	}

/////////////////////////////////////////////////////////////////////////////////
// Video - wrapper over the VideoImpl class

Video::Video(const char* filename, const char *open_flags)
	: 	m_impl(new Private::VideoImpl(getBOption("verbose"), filename, open_flags)),
		m_pixmap(0)
{
	addIOption("width", -1, "output width");
	addIOption("height", -1, "output height");
	addFOption("bitrate", 150000, "output bitrate");
	addFOption("framerate", 25, "output framerate");
	addIOption("gop", 100, "output gop size");
}
Video::~Video()
{
	delete m_impl;
	delete[] m_pixmap;
}

bool Video::open(const char* filename, const char *open_flags)
{
	if (m_impl->open(getBOption("verbose"), filename, open_flags) == true)
	{
		// Make sure the read statistics are set to the main Video class
		const int w = m_impl->getWidth(), h = m_impl->getHeight(), gop = m_impl->getGop();
		const float bitrate = m_impl->getBitrate(), framerate = m_impl->getFramerate();
		setIOption("width", w);
		setIOption("height", h);
		setFOption("bitrate", bitrate);
		setFOption("framerate", framerate);
		setIOption("gop", gop);
		delete[] m_pixmap;
		m_pixmap = new unsigned char[3 * w * h];
		return true;
	}
	return false;
}
void Video::close()
{
	m_impl->close();
}

bool Video::read(Image& image)
{
	if (m_impl->read(getBOption("verbose")) == false)
		return false;

	// OK: copy the pixmap to the image
	image.resize(m_impl->getWidth(), m_impl->getHeight(), image.getNPlanes());
	Image::fillImage(m_impl->getPixmap(), 3, image);
	return true;
}
bool Video::write(const Image& image)
{
	// Copy the image to the pixmap
	if (	m_pixmap == 0 ||
		image.getWidth() != m_impl->getWidth() ||
		image.getHeight() != m_impl->getHeight())
	{
		return false;
	}
	Image::fillPixmap(m_pixmap, 3, image);

	return m_impl->write(getBOption("verbose"), m_pixmap);
}

const char* Video::codec() const
{
	return m_impl->codec();
}

int Video::getNFrames() const
{
	return m_impl->getNFrames();
}
Video::State Video::getState() const
{
	return m_impl->getState();
}

void Video::optionChanged(const char* name)
{
	m_impl->setup(	getIOption("width"),
			getIOption("height"),
			getFOption("bitrate"),
			getFOption("framerate"),
			getIOption("gop"));
}

/////////////////////////////////////////////////////////////////////////////////

}
