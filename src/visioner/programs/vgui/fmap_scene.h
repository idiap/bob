/**
 * @file visioner/programs/vgui/fmap_scene.h
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

#ifndef FMAP_SCENE_H
#define FMAP_SCENE_H

#include <QGraphicsScene>
#include <QAction>
#include <QMenu>

#include "fmap_item.h"

/**
 * FeatureMapScene:
 *	- collection of various FeatureMapItem objects
 */

class FeatureMapScene : public QGraphicsScene
{
  Q_OBJECT

  public:

    // Constructor
    FeatureMapScene(const SceneSettings& settings, QWidget* parent = 0);

    // Destructor
    ~FeatureMapScene();

    // Update with the items' position and settings
    void updateSettings();

    // Access functions 
    const SceneSettings& settings() const { return m_settings; }
    void setSettings(const SceneSettings& settings);

    static int sceneWidth() { return 4800; }
    static int sceneHeight() { return 3600; }
    static int border() { return 8; }

  protected:

    // Drawing & events
    void drawBackground(QPainter* painter, const QRectF& rect);
    void contextMenuEvent(QGraphicsSceneContextMenuEvent* event);
    void keyPressEvent(QKeyEvent* event);
    void mousePressEvent(QGraphicsSceneMouseEvent* event);

    private slots:

      // Contextual menu events
      void onDrawSourceChanged(QAction* action);
    void onDrawModeChanged(QAction* action);
    void onNewClicked();
    void onCopyClicked();
    void onDeleteSelectedClicked();
    void onDeleteAllClicked();
    void onSaveItemClicked();
    void onSaveSceneClicked();
    void onSettingsItemClicked();
    void onSettingsSceneClicked();

    // Item management events
    void mySelectionChanged();

  private:

    // Item management	
    void add(bool asCopy = false);
    void erase();
    void clear();
    int noItems() const { return items().size(); }

  private:

    // Manage item selection
    QList<FeatureMapItem*> myItems() const;
    FeatureMapItem* mySelectedItem() const;

    // Visibile region in screen coordinates
    QRectF visibleRect() const;

  private:

    // Attributes
    SceneSettings		m_settings;	// Drawing settings for the whole container

    QMenu*			m_menu_draw_source;
    QMenu*			m_menu_draw_mode;
    QAction*		m_actionDrawSource[DrawingSource_End];
    QAction*		m_actionDrawMode[DrawingMode_End];
    QAction*		m_actionNew;
    QAction*		m_actionCopy;
    QAction*		m_actionDeleteSelected;
    QAction*		m_actionDeleteAll;
    QAction*		m_actionSaveItem;
    QAction*		m_actionSaveScene;
    QAction*		m_actionSettingsItem;
    QAction*		m_actionSettingsScene;
};

#endif
