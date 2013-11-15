/**
 * @file visioner/programs/vgui/main.cc
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include <QtGui/QApplication>
#include <QtGui/QSplashScreen>

#include "fmap_widget.h"

int main(int argc, char *argv[]) {

  QApplication app(argc, argv);
  QApplication::setApplicationName("Visioner-GUI");
  QApplication::setWindowIcon(QIcon(":/icons/mainframeSmall.png"));

  // Create splash widget
  QSplashScreen* splash = new QSplashScreen;
  splash->setPixmap(QPixmap(":/icons/mainframeBig.png"));
  splash->setFont(QFont("Times", 10, QFont::Bold));
  splash->setWindowOpacity(1.0);
  splash->show();

  // Load the application
  splash->showMessage(QObject::tr("Loading GUI ..."),
      Qt::AlignRight | Qt::AlignBottom, Qt::blue);
  FeatureMapWidget widget;
  widget.showMaximized();

  // Delete the splash widget
  splash->finish(&widget);
  delete splash;
  widget.showMaximized();

  return app.exec();

}
