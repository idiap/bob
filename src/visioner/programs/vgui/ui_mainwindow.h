/**
 * @file visioner/programs/vgui/ui_mainwindow.h
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

/**
 * Form generated from reading UI file 'mainwindow.ui'
 *
 * Created: Tue Apr 13 17:19:03 2010
 * by: Qt User Interface Compiler version 4.6.2
 *
 * WARNING! All changes made in this file will be lost when recompiling it
 */

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QHeaderView>
#include <QtGui/QMainWindow>
#include <QtGui/QMenuBar>
#include <QtGui/QStatusBar>
#include <QtGui/QToolBar>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
  public:
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QWidget *centralWidget;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
      if (MainWindow->objectName().isEmpty())
        MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
      MainWindow->resize(600, 400);
      menuBar = new QMenuBar(MainWindow);
      menuBar->setObjectName(QString::fromUtf8("menuBar"));
      MainWindow->setMenuBar(menuBar);
      mainToolBar = new QToolBar(MainWindow);
      mainToolBar->setObjectName(QString::fromUtf8("mainToolBar"));
      MainWindow->addToolBar(mainToolBar);
      centralWidget = new QWidget(MainWindow);
      centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
      MainWindow->setCentralWidget(centralWidget);
      statusBar = new QStatusBar(MainWindow);
      statusBar->setObjectName(QString::fromUtf8("statusBar"));
      MainWindow->setStatusBar(statusBar);

      retranslateUi(MainWindow);

      QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
      MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
  class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
