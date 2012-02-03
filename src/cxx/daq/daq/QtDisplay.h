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
