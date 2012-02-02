#include "daq/QtDisplay.h"

namespace bob { namespace daq {

static pthread_mutex_t pthread_mutex_initializer = PTHREAD_MUTEX_INITIALIZER;

QtDisplay::QtDisplay() : QWidget() {
  init();
}

QtDisplay::QtDisplay(QWidget *parent, Qt::WindowFlags f) : QWidget(parent, f) {
  init();
}

void QtDisplay::init() {
  img_mutex = pthread_mutex_initializer;
  boundingBox_mutex = pthread_mutex_initializer;
  boundingBox.detected = false;

  isRecording = false;

  connect(this, SIGNAL(closeInGuiThread()), this, SLOT(close()), Qt::QueuedConnection);
}

QtDisplay::~QtDisplay() {

}

void QtDisplay::start() {
  isRecording = false;
  captureStatus.recordingDelay = 0;
  captureStatus.elapsedTime = 0;
  captureStatus.frameNb = -1;
  captureStatus.totalSessionTime = -1;
  captureStatus.isRecording = false;

  if (!thumbnail.empty()) {
    if (thumnailImage.load(thumbnail.c_str())) {
      int max_size = 100;

      if (thumnailImage.width() > thumnailImage.height()) {
        thumnailRect = QRect(0, 0, max_size, (thumnailImage.height() * max_size) / thumnailImage.width());
      }
      else {
        thumnailRect = QRect(0, 0, (thumnailImage.width() * max_size) / thumnailImage.height(), max_size);
      }
    }
  }
  
  
  QCoreApplication* app = QApplication::instance();

  if (fullscreen) {
    //this->showFullScreen();
    this->showMaximized();
  }
  else {
    if (displayHeight > 0 && displayWidth > 0) {
      this->resize(displayWidth, displayHeight);
    }
    
    this->show();
  }
  
  app->exec();

  isRecording = false;
  if (!onStopRecording.empty()) {
    system(onStopRecording.c_str());
  }
}

void QtDisplay::stop() {
  //this->close();
  emit closeInGuiThread();
}

void QtDisplay::imageReceived(blitz::Array<unsigned char, 2>& image, CaptureStatus& status) {
  pthread_mutex_lock(&img_mutex);
  //img = cv::Mat(cv::Size(image.cols() / 3, image.rows()), CV_8UC3, image.data()).clone();
  img = QByteArray((char*)image.data(), image.size());
  imgSize = QSize(image.cols() / 3, image.rows());
  this->captureStatus = status;
  pthread_mutex_unlock(&img_mutex);
  
  this->update();
}


void QtDisplay::onDetection(FaceLocalizationCallback::BoundingBox& bb) {
  pthread_mutex_lock(&boundingBox_mutex);
  boundingBox = bb;
  pthread_mutex_unlock(&boundingBox_mutex);
}

void QtDisplay::paintEvent(QPaintEvent* event) {
  QPainter painter(this);

  QByteArray ba;
  QSize imageSize;

  CaptureStatus status;
  
  pthread_mutex_lock(&img_mutex);
  if (!img.isEmpty()) {
    char header[255] = {0};
    imageSize = imgSize;
    sprintf(header, "P6\n%d %d\n255\n", imgSize.width(), imgSize.height());
    ba.append(header);
    ba.append(img);
  }
  
  status = captureStatus;
  pthread_mutex_unlock(&img_mutex);

  if (isRecording != status.isRecording) {
    isRecording = status.isRecording;

    if (status.isRecording) {
      if (!onStartRecording.empty()) {
        system(onStartRecording.c_str());
      }
    }
  }
    
  QRect canvas = painter.window();
  if (!ba.isEmpty()) {
    if (status.frameNb < 2 && (displayHeight < 0 || displayWidth < 0)) {
      this->resize(imageSize);
    }
    
    QPixmap p;
    if (p.loadFromData(ba)) {
      painter.drawPixmap(canvas, p);
    }

    if (!thumnailImage.isNull()) {
      painter.drawPixmap(thumnailRect, thumnailImage);
    }
  
    FaceLocalizationCallback::BoundingBox bbx;
    pthread_mutex_lock(&boundingBox_mutex);
    if (boundingBox.detected) {
      bbx = boundingBox;
    }
    pthread_mutex_unlock(&boundingBox_mutex);

    if (bbx.detected) {
      float xratio = canvas.width() / (float)imageSize.width();
      float yratio = canvas.height() / (float)imageSize.height();

      bbx.x *= xratio;
      bbx.y *= yratio;
      bbx.width *= xratio;
      bbx.height *= yratio;

      painter.drawRect(bbx.x, bbx.y, bbx.width, bbx.height);
    }

    // Compute usefull times
    double delay_remaining = status.recordingDelay - status.elapsedTime;
    double total_remaining = status.totalSessionTime - status.elapsedTime;

    // Things to do before recoring
    if (delay_remaining > 0) {
      
    }

    // Things to do when recording
    if (total_remaining > 0 && status.totalSessionTime > 0) {
      // Display progress bar
      double fract = (status.totalSessionTime  - total_remaining) / status.totalSessionTime;
      painter.fillRect(QRect(QPoint(canvas.width() * fract, canvas.height() - 10), canvas.bottomRight()), Qt::blue);
      

      if (delay_remaining > 0) {
        double fract_delay = 1 - (status.totalSessionTime  - status.recordingDelay) / status.totalSessionTime;
        painter.fillRect(QRect(QPoint(canvas.width() * fract, canvas.height() - 10),
                               QPoint(canvas.width() * fract_delay, canvas.height())), Qt::red);
      }
    }
    
    // Display text
    
    if (!text.empty()) {
      QFont myfont = painter.font();
      myfont.setBold(true);
      myfont.setPixelSize(myfont.pointSize() + 10);

      painter.setFont(myfont);
      
      QString customText(text.c_str());
      int textWidth = painter.fontMetrics().width(customText);
      int textHeight = painter.fontMetrics().height();

      painter.drawText(canvas.width() / 2 - textWidth / 2, textHeight + 5, customText);
    }

    // OpenCVDisplay recording circle
    if (status.isRecording) {
      painter.setBrush(QBrush(Qt::red));
      painter.drawEllipse(canvas.width() - 25, 15, 10, 10);
    }
    else {
      painter.setBrush(QBrush(Qt::gray));
      painter.drawEllipse(canvas.width() - 25, 15, 10, 10);
    }
  }
  
}

void QtDisplay::keyPressEvent(QKeyEvent *keyEvent) {
  if (!keyEvent->text().isEmpty()) {
    pthread_mutex_lock(&callbacks_mutex);
    for(std::vector<KeyPressCallback*>::iterator it = callbacks.begin(); it != callbacks.end(); it++) {
      int key = keyEvent->text()[0].toAscii();
      (*it)->keyPressed(key);
    }
    pthread_mutex_unlock(&callbacks_mutex);
  }
}
  
}}