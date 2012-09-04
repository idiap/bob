/**
 * @file visioner/programs/vgui/fmap_item.h
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

#ifndef FMAP_ITEM_H
#define FMAP_ITEM_H

#include <QGraphicsItem>

#include "bob/visioner/util/matrix.h"
#include "bob/visioner/util/histogram.h"
#include "bob/visioner/vision/vision.h"

#include "settings.h"

/**
 * FeatureMapItem:
 *	- draws various feature maps or histograms of feature maps
 *	- can be drawn, resized, scaled, panned ...
 */
class FeatureMapItem : public QGraphicsItem {

  public:

    // Constructor
    FeatureMapItem(const SceneSettings& global, const ItemSettings& settings);

    // Bounding rectangle
    QRectF boundingRect() const;

    // Access functions
    const ItemSettings& settings() const { return m_settings; }

    void setSettings(const ItemSettings& settings);
    bool setSource(DrawingSource source);
    bool setMode(DrawingMode mode);

    void setSize(qreal width, qreal height);

  protected:

    // Drawing & events
    void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
    void mousePressEvent(QGraphicsSceneMouseEvent* event);
    void mouseMoveEvent(QGraphicsSceneMouseEvent* event);
    void mouseReleaseEvent(QGraphicsSceneMouseEvent* event);

  private:

    // Update the feature map
    bool update_fmap(const ItemSettings& settings);
    void update_fmap();

    // Compute the feature map / drawing area
    QRectF imageRect() const;
    QRectF drawingRect() const;

    // Draw the feature map
    void drawMap(QPainter& painter);

    // Draw the ground truth
    void drawGTruth(QPainter& painter);

    // Draw the histogram of the selected feature map
    void drawHistogram(QPainter& painter);

    // Draw the frame with the text infos
    void drawTextInfos(QPainter& painter);

    // Draw a 2D graphic for some stored values
    void draw2DPlot(QPainter& painter, double value_min, double value_max,
        const std::vector<double>& values);

    // Draw an arrow between two points
    void drawArrow(QPainter& painter, int startx, int starty, int stopx, int stopy);

  private:

    enum MouseType
    {
      Ignore,
      BottomLeftResize,
      BottomRightResize,
      TopLeftResize,
      TopRightResize,
      Pane,
      Zoom
    };

    // Attributes
    const SceneSettings&		m_global;	// Drawing settings for the whole scene
    ItemSettings			m_settings;	// Drawing settings for the item

    bob::visioner::Matrix<uint32_t>		m_src_image;	// Source data: image
    bob::visioner::Matrix<uint32_t>		m_src_iimage;	// Source data: integral image
    bob::visioner::Matrix<uint16_t>	m_src_fmap;	// Source data: feature map
    QImage				m_src_qfmap;	// Source data: feature map transformed to QImage
    bob::visioner::Histogram		m_src_histo;	// Source data: feature map histogram
    const std::vector<QRgb>*	m_src_colors;	// Value to RGB mapping

    MouseType			m_mouse;	// Current mouse action

    float				m_pane_x, m_pane_y;// Paning coeficients in both directions
};

#endif
