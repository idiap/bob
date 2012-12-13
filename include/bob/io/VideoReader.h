/**
 * @file bob/io/VideoReader.h
 * @date Mon 10 Dec 2012 14:53:08 CET
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief A switch between different versions of ffmpeg/libav
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef BOB_IO_VIDEOREADER_H
#define BOB_IO_VIDEOREADER_H

extern "C" {
#include <libavcodec/avcodec.h>
}

#if LIBAVCODEC_VERSION_INT < 0x350700 //53.7.0 @ ffmpeg 0.8
#include "bob/io/VideoReader1.h"
#else
#include "bob/io/VideoReader2.h"
#endif

#endif /* BOB_IO_VIDEOREADER_H */
