/**
 * @file cxx/daq/daq/ConsoleDisplay.h
 * @date Thu Feb 2 11:22:57 2012 +0100
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
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
#ifndef CONSOLEDISPLAY_H
#define CONSOLEDISPLAY_H

#include <daq/Display.h>

namespace bob { namespace daq {

/**
 * Dispay class that prints a console message when a frame or a detection is
 * received.
 */
class ConsoleDisplay : public bob::daq::Display {
public:
  ConsoleDisplay();
  virtual ~ConsoleDisplay();
  
  void stop();
  void start();
  void imageReceived(blitz::Array<unsigned char, 2>& image, CaptureStatus& status);
  void onDetection(BoundingBox& bb);

private:
  pthread_mutex_t mutex;
  pthread_cond_t cond;

  bool mustStop;
};

}}

#endif // CONSOLEDISPLAY_H
