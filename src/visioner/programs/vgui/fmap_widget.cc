/**
 * @file visioner/programs/vgui/fmap_widget.cc
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
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

#include "fmap_widget.h"
#include <QSplitter>
#include <QDataStream>
#include "controls.h"
#include "image_collection.h"

static ImageCollection& theImages = ImageCollection::get_mutable_instance();

  FeatureMapWidget::FeatureMapWidget(QWidget* parent)
:	QWidget(parent),
  m_scene(0),
  m_timerSlide(0)
{
  // Load state
  QFile file("vgui.state");
  file.open(QIODevice::ReadOnly);
  if (file.isOpen())
  {
    QDataStream data(&file);

    // Image collection
    std::vector<std::string> listfiles;
    std::size_t index;
    data >> listfiles;
    data >> index;

    theImages.clear();
    for (std::vector<std::string>::const_iterator it = listfiles.begin(); it != listfiles.end(); ++ it)
    {
      theImages.add(*it);
    }
    theImages.move(index);

    // Scene
    SceneSettings settings;
    data >> settings;
    m_scene = new FeatureMapScene(settings, this);
  }
  else
  {
    m_scene = new FeatureMapScene(SceneSettings(), this);
  }

  // Initialize the controls
  m_buttonClearImages = pushButton("Clear", "", "Delete the image & ground truth files");
  m_buttonLoadImages = pushButton("Browse ...", "", "Load image & ground truth files");	

  m_listImages = new QListWidget(this);
  m_listImages->setSelectionMode(QAbstractItemView::SingleSelection);

  m_labelImages = new QLabel("");
  m_labelImages->setFrameStyle(QFrame::StyledPanel);

  m_buttonSlidePrev = toolButton(":/icons/previous.png", "Previous image");
  m_buttonSlideRewind = toolButton(":/icons/rewind.png", "Rewind the slideshow");
  m_buttonSlideStart = toolButton(":/icons/start.png", "Start the slideshow");
  m_buttonSlidePause = toolButton(":/icons/pause.png", "Pause the slideshow");
  m_buttonSlideNext = toolButton(":/icons/next.png", "Next image");

  // Assembly the controls	
  assemblyControls();

  // Connect controls
  connect(m_buttonClearImages, SIGNAL(clicked()), this, SLOT(onClearImages()));
  connect(m_buttonLoadImages, SIGNAL(clicked()), this, SLOT(onLoadImages()));
  connect(m_listImages, SIGNAL(currentRowChanged(int)), this, SLOT(onSelectImage(int)));
  connect(m_listImages, SIGNAL(clicked(const QModelIndex&)), this, SLOT(onSelectImage(const QModelIndex&)));

  connect(m_buttonSlidePrev, SIGNAL(clicked()), this, SLOT(onSlidePrev()));
  connect(m_buttonSlideRewind, SIGNAL(clicked()), this, SLOT(onSlideRewind()));
  connect(m_buttonSlideStart, SIGNAL(clicked()), this, SLOT(onSlideStart()));
  connect(m_buttonSlidePause, SIGNAL(clicked()), this, SLOT(onSlidePause()));
  connect(m_buttonSlideNext, SIGNAL(clicked()), this, SLOT(onSlideNext()));

  populateImageList();

  m_timerSlide = new QTimer(this);
  QObject::connect(m_timerSlide, SIGNAL(timeout()), this, SLOT(onSlideTimer()));	
}

FeatureMapWidget::~FeatureMapWidget()
{
  // Save state
  QFile file("vgui.state");
  file.open(QIODevice::WriteOnly);
  if (file.isOpen())
  {
    QDataStream data(&file);

    // Image collection
    data << theImages.listfiles();
    data << theImages.index();

    // Scene
    m_scene->updateSettings();
    data << m_scene->settings();
  }

  // Cleanup
  delete m_scene;
  delete m_timerSlide;	
}

void FeatureMapWidget::assemblyControls()
{
  // Left - image collection controls
  QGridLayout* left_layout = new QGridLayout();
  {
    int row = 0;
    left_layout->addWidget(new QLabel("Manage: "), row, 0, 1, 1);
    left_layout->addWidget(m_buttonClearImages, row, 1, 1, 1);
    left_layout->addWidget(m_buttonLoadImages, row, 2, 1, 1);		
    row ++;
    left_layout->addWidget(m_labelImages, row, 0, 1, 3);
    row ++;
    left_layout->addWidget(m_listImages, row, 0, 1, 3);
  }

  // Right - toolbar controls + scene
  QVBoxLayout* right_layout = new QVBoxLayout();
  {
    // Build the toolbar
    QHBoxLayout* layout = new QHBoxLayout;
    layout->addWidget(m_buttonSlidePrev);
    layout->addWidget(m_buttonSlideRewind);
    layout->addWidget(m_buttonSlideStart);
    layout->addWidget(m_buttonSlidePause);
    layout->addWidget(m_buttonSlideNext);
    layout->addStretch();

    // Scene
    QGraphicsView* view = new QGraphicsView;
    m_scene->setItemIndexMethod(QGraphicsScene::NoIndex);
    //view->setDragMode(QGraphicsView::ScrollHandDrag);
    view->setScene(m_scene);
    view->setCacheMode(QGraphicsView::CacheBackground);
    view->setViewportUpdateMode(QGraphicsView::BoundingRectViewportUpdate);
    view->setRenderHint(QPainter::Antialiasing);
    view->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    view->scale(qreal(1.0), qreal(1.0));
    view->setMinimumSize(240, 180);
    view->setWindowTitle(tr("Elastic Nodes"));

    right_layout->addLayout(layout, 0);
    right_layout->addWidget(view, 1000);
  }

  // Main layout
  QWidget* left_widget = new QWidget();
  left_widget->setLayout(left_layout);

  QWidget* right_widget = new QWidget();
  right_widget->setLayout(right_layout);

  QSplitter* splitter = new QSplitter(Qt::Horizontal);
  splitter->addWidget(left_widget);
  splitter->addWidget(right_widget);
  splitter->setStretchFactor(0, 5);
  splitter->setStretchFactor(1, 95);

  QVBoxLayout* main_layout = new QVBoxLayout();
  main_layout->addWidget(splitter);
  setLayout(main_layout);
}

void FeatureMapWidget::resizeEvent(QResizeEvent* event)
{
  QWidget::resizeEvent(event);
  updateControls();
}

void FeatureMapWidget::updateScene()
{
  m_scene->invalidate();
}

void FeatureMapWidget::updateControls()
{
  // Update the current image
  if (	m_listImages->currentRow() != (int)theImages.index())
  {
    m_listImages->setCurrentRow(theImages.index());
  }

  // Update the slideshow controls
  m_buttonSlidePrev->setEnabled(theImages.index() > 0 && !m_timerSlide->isActive());
  m_buttonSlideRewind->setEnabled(!theImages.empty() && !m_timerSlide->isActive());
  m_buttonSlideStart->setEnabled(!theImages.empty() && !m_timerSlide->isActive());
  m_buttonSlidePause->setEnabled(!theImages.empty() && m_timerSlide->isActive());
  m_buttonSlideNext->setEnabled(theImages.index() + 1 < theImages.size() && !m_timerSlide->isActive());

  // Update the status labels
  m_labelImages->setText(QObject::tr("Collection: %1/%2 images\nImage name: <%3>\nImage size: %4x%5 pixels")
      .arg(theImages.index()).arg(theImages.size())
      .arg(theImages.name().c_str())
      .arg(theImages.ipscale().cols()).arg(theImages.ipscale().rows()));
}

void FeatureMapWidget::onClearImages()
{
  theImages.clear();
  m_listImages->setCurrentRow(-1);
  m_listImages->clearSelection();
  m_listImages->setSelectionMode(QAbstractItemView::NoSelection);
  while (m_listImages->count() > 0)
  {
    m_listImages->takeItem(0);
  }

  updateControls();
}

void FeatureMapWidget::onLoadImages()
{
  QString filename = QFileDialog::getOpenFileName(this,
      "Load image & ground truth list",
      "./",
      "List files (*.list)");
  if (filename.isNull() || filename.isEmpty())
  {
    return;
  }

  if (theImages.add(filename.toStdString()) == false)
  {
    QMessageBox::warning(this, "ERROR", "Failed to load the file list!");
    return;
  }

  populateImageList();
  updateControls();
}

void FeatureMapWidget::populateImageList()
{
  m_listImages->setCurrentRow(-1);
  m_listImages->clearSelection();
  m_listImages->setSelectionMode(QAbstractItemView::NoSelection);
  while (m_listImages->count() > 0)
  {
    m_listImages->takeItem(0);
  }
  for (std::vector<std::string>::const_iterator it_i = theImages.ifiles().begin(),
      it_g = theImages.gfiles().begin(); it_i != theImages.ifiles().end(); ++ it_i, ++ it_g)
  {
    m_listImages->insertItem(
        m_listImages->count(),
        new QListWidgetItem(QFileInfo(it_i->c_str()).fileName() 
          + " ~ " + QFileInfo(it_g->c_str()).fileName()));
  }
  m_listImages->setSelectionMode(QAbstractItemView::SingleSelection);
}

void FeatureMapWidget::onSelectImage(int currentRow)
{
  theImages.move(currentRow);
  updateControls();
  updateScene();
}

void FeatureMapWidget::onSelectImage(const QModelIndex& index)
{
  onSelectImage(index.row());
}

void FeatureMapWidget::onSlidePrev()
{
  theImages.previous();
  updateControls();
  updateScene();
}

void FeatureMapWidget::onSlideRewind()
{
  theImages.rewind();
  updateControls();
  updateScene();
}

void FeatureMapWidget::onSlideStart()
{
  m_timerSlide->start(1000);
}

void FeatureMapWidget::onSlidePause()
{
  m_timerSlide->stop();
  updateControls();
}

void FeatureMapWidget::onSlideNext()
{
  theImages.next();
  updateControls();
  updateScene();
}

void FeatureMapWidget::onSlideTimer()
{
  if (theImages.next())
  {
    updateControls();
    updateScene();
  }
  else
  {
    m_timerSlide->stop();
  }
}
