/**
 * @file cxx/daq/daq/QtDisplay.h
 * @date Thu Feb 2 11:22:57 2012 +0100
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 *
 * Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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
#ifndef QTDISPLAY_H
#define QTDISPLAY_H

#include <daq/Display.h>
#include <QtGui>


namespace bob { namespace daq {
  
/**
 * Display a GUI using Qt
 */
class QtDisplay: public QWidget, public Display {
  Q_OBJECT
  
public:
  QtDisplay();
  QtDisplay(QWidget* parent, Qt::WindowFlags f = 0);
  virtual ~QtDisplay();

  void imageReceived(blitz::Array<unsigned char, 2>& image, CaptureStatus& status);
  void onDetection(FaceLocalizationCallback::BoundingBox& bb);
  
  void start();
  void stop();

signals:
  void closeInGuiThread();
  
protected:
  virtual void keyPressEvent(QKeyEvent* keyEvent);
  virtual void paintEvent(QPaintEvent* event);


private:
  void init();
  
  QByteArray img;
  pthread_mutex_t img_mutex;
  CaptureStatus captureStatus;
  
  FaceLocalizationCallback::BoundingBox boundingBox;
  pthread_mutex_t boundingBox_mutex;

  QSize imgSize;

  QPixmap thumnailImage;
  QRect thumnailRect;

  bool isRecording;
};

}}

#endif // QTDISPLAY_H
