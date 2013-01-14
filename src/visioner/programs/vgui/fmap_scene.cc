/**
 * @file visioner/programs/vgui/fmap_scene.cc
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

#include "fmap_scene.h"
#include "controls.h"
#include "settings_dialog.h"

FeatureMapScene::FeatureMapScene(const SceneSettings& settings, QWidget* parent)
	:	QGraphicsScene(parent),
		m_settings(settings)
{
	setSceneRect(-sceneWidth() / 2, -sceneHeight() / 2, sceneWidth(), sceneHeight());
	
	// Initialize items as given by the configuration
	for (std::vector<ItemSettings>::iterator it = m_settings.m_items.begin();
		it != m_settings.m_items.end(); ++ it)
	{
		FeatureMapItem* item = new FeatureMapItem(m_settings, *it);
		addItem(item);
		
		item->setSelected(it + 1 == m_settings.m_items.end());
	}
		
	// Initialize contextual menu actions
	m_menu_draw_source = new QMenu(tr("&Draw source"));
	for (int i = DrawingSource_Begin; i < DrawingSource_End; i ++)
	{
		m_actionDrawSource[i] = action("", DrawingSourceStr[i], true);
		m_menu_draw_source->addAction(m_actionDrawSource[i]);
	}

	m_menu_draw_mode = new QMenu(tr("&Draw mode"));
	for (int i = DrawingMode_Begin; i < DrawingMode_End; i ++)
	{
		m_actionDrawMode[i] = action("", DrawingModeStr[i], true);
		m_menu_draw_mode->addAction(m_actionDrawMode[i]);
	}

	m_actionNew = action(":/icons/new.png", "New\tCtrl+N", false);
	m_actionCopy = action(":/icons/copy.png", "Copy\tCtrl+C", false);
	m_actionDeleteSelected = action(":/icons/delete.png", "Delete selected\tDel", false);
	m_actionDeleteAll = action(":/icons/clear.png", "Delete all\tShift+Del", false);
	m_actionSaveItem = action(":/icons/save.png", "Save item\tCtrl+S", false);
	m_actionSaveScene = action(":/icons/save.png", "Save scene\tCtrl+Shift+S", false);
	m_actionSettingsItem = action(":/icons/settings.png", "Settings item\tCtrl+K", false);
	m_actionSettingsScene = action(":/icons/settings.png", "Settings scene\tCtrl+Shift+K", false);

	// Connect contextual menu actions
	connect(m_menu_draw_source, SIGNAL(triggered(QAction*)), this, SLOT(onDrawSourceChanged(QAction*)));
	connect(m_menu_draw_mode, SIGNAL(triggered(QAction*)), this, SLOT(onDrawModeChanged(QAction*)));
	connect(m_actionNew, SIGNAL(triggered()), this, SLOT(onNewClicked()));
	connect(m_actionCopy, SIGNAL(triggered()), this, SLOT(onCopyClicked()));
	connect(m_actionDeleteSelected, SIGNAL(triggered()), this, SLOT(onDeleteSelectedClicked()));
	connect(m_actionDeleteAll, SIGNAL(triggered()), this, SLOT(onDeleteAllClicked()));
	connect(m_actionSaveItem, SIGNAL(triggered()), this, SLOT(onSaveItemClicked()));
	connect(m_actionSaveScene, SIGNAL(triggered()), this, SLOT(onSaveSceneClicked()));
	connect(m_actionSettingsItem, SIGNAL(triggered()), this, SLOT(onSettingsItemClicked()));
	connect(m_actionSettingsScene, SIGNAL(triggered()), this, SLOT(onSettingsSceneClicked()));

	connect(this, SIGNAL(selectionChanged()), this, SLOT(mySelectionChanged()));
}

FeatureMapScene::~FeatureMapScene()
{
	for (int i = DrawingSource_Begin; i < DrawingSource_End; i ++)
	{
		delete m_actionDrawSource[i];
	}
	for (int i = DrawingMode_Begin; i < DrawingMode_End; i ++)
	{
		delete m_actionDrawMode[i];
	}

	delete m_actionNew;
	delete m_actionCopy;
	delete m_actionDeleteSelected;
	delete m_actionDeleteAll;
	delete m_actionSaveItem;
	delete m_actionSaveScene;
	delete m_actionSettingsItem;
	delete m_actionSettingsScene;

	delete m_menu_draw_source;
	delete m_menu_draw_mode;
}

void FeatureMapScene::setSettings(const SceneSettings&settings)
{
	m_settings = settings;
	invalidate(sceneRect(), QGraphicsScene::BackgroundLayer);
}

void FeatureMapScene::drawBackground(QPainter* painter, const QRectF& rect)
{
	m_settings.m_frame.drawBackground(*painter, sceneRect());
	m_settings.m_frame.drawBorder(*painter, sceneRect());
}

void FeatureMapScene::contextMenuEvent(QGraphicsSceneContextMenuEvent* event)
{	
//	// TODO: select the item under the right click - if any!
//	http://www.qtcentre.org/threads/43935-how-not-ta-clear-the-selection-on-QGraphicsScene-right-mouse-press-event
//	{
//		QGraphicsItem* item = itemAt(event->pos());
//		if (item != 0)
//		{
//			std::cout << "contextMenuEvent\n";
//			if (item != selectedItems()[0])
//				std::cout << "not the selected one!\n";
//			item->setSelected(true);
//			selectionChanged();
//		}
//	}

	// Build the contextual menu and display it
	event->accept();

	const FeatureMapItem* item = mySelectedItem();

	QMenu menu;

	// Item-related menu
	if (itemAt(event->scenePos()) != 0)
	{
		//	drawing settings
		menu.addMenu(m_menu_draw_source);
		menu.addMenu(m_menu_draw_mode);

		m_menu_draw_source->setEnabled(item != 0);
		m_menu_draw_mode->setEnabled(item != 0);

		if (item != 0)
		{
			for (int i = DrawingSource_Begin; i < DrawingSource_End; i ++)
			{
				m_actionDrawSource[i]->setChecked(item->settings().m_source == i);
			}
			for (int i = DrawingMode_Begin; i < DrawingMode_End; i ++)
			{
				m_actionDrawMode[i]->setChecked(item->settings().m_mode == i);
			}
		}

		//	saving content as image
		menu.addSeparator();
		menu.addAction(m_actionSaveItem);
		m_actionSaveItem->setEnabled(item != 0);

		//	settings
		menu.addAction(m_actionSettingsItem);
		m_actionSettingsItem->setEnabled(item != 0);
	}

	// Scene-related menu
	else
	{
		//	item management
		menu.addAction(m_actionNew);
		menu.addAction(m_actionCopy);
		menu.addAction(m_actionDeleteSelected);
		menu.addAction(m_actionDeleteAll);

		m_actionCopy->setEnabled(item != 0);
		m_actionDeleteSelected->setEnabled(selectedItems().size() > 0);
		m_actionDeleteAll->setEnabled(items().size() > 0);

		//	saving content as image
		menu.addSeparator();
		menu.addAction(m_actionSaveScene);

		//	settings
		menu.addAction(m_actionSettingsScene);
	}

	// OK, display the menu
	menu.exec(event->screenPos());
}

void FeatureMapScene::keyPressEvent(QKeyEvent* event)
{
	// Ctrl+C - create a copy of the selected item
	if (	(event->modifiers() & Qt::ControlModifier) == Qt::ControlModifier &&
		event->key() == Qt::Key_C)
	{
		onCopyClicked();
	}

	// Ctrl+N - create a new default item
	else if (	(event->modifiers() & Qt::ControlModifier) == Qt::ControlModifier &&
			event->key() == Qt::Key_N)
	{
		onNewClicked();
	}

	// Shift+Del - delete the unselected items
	else if (	(event->modifiers() & Qt::ShiftModifier) == Qt::ShiftModifier &&
			event->key() == Qt::Key_Delete)
	{
		onDeleteAllClicked();
	}

	// Del - delete the selected item
	else if (event->key() == Qt::Key_Delete)
	{
		onDeleteSelectedClicked();
	}

	// Ctrl+Shift+S - save the scene
	else if (	(event->modifiers() & Qt::ControlModifier) == Qt::ControlModifier &&
			(event->modifiers() & Qt::ShiftModifier) == Qt::ShiftModifier &&
			event->key() == Qt::Key_S)
	{
		onSaveSceneClicked();
	}

	// Ctrl+S - save the selected item
	else if (	(event->modifiers() & Qt::ControlModifier) == Qt::ControlModifier &&
			event->key() == Qt::Key_S)
	{
		onSaveItemClicked();
	}

	// Ctrl+Shift+K - change the settings of the scene
	else if (	(event->modifiers() & Qt::ControlModifier) == Qt::ControlModifier &&
			(event->modifiers() & Qt::ShiftModifier) == Qt::ShiftModifier &&
			event->key() == Qt::Key_K)
	{
		onSettingsSceneClicked();
	}

	// Ctrl+K - change the settings of the selected item
	else if (	(event->modifiers() & Qt::ControlModifier) == Qt::ControlModifier &&
			event->key() == Qt::Key_K)
	{
		onSettingsItemClicked();
	}
        
        // Increase/decrease cell size
        else if (       event->key() == Qt::Key_Up ||
                        event->key() == Qt::Key_Down ||
                        event->key() == Qt::Key_Left ||
                        event->key() == Qt::Key_Right)
        {
                FeatureMapItem* item = mySelectedItem();
                if (item != 0)
                {
                        ItemSettings settings = item->settings();                        
                        
                        if (    (event->key() == Qt::Key_Up && settings.inc_cy() == true) ||
                                (event->key() == Qt::Key_Down && settings.dec_cy() == true) ||
                                (event->key() == Qt::Key_Right && settings.inc_cx() == true) ||
                                (event->key() == Qt::Key_Left && settings.dec_cx() == true))
                        {
                                item->setSettings(settings);
                                updateSettings();
                        }
                }
        }
}

void FeatureMapScene::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
	QGraphicsScene::mousePressEvent(event);
	
	// If an item is selected, make sure it is on top
	if (itemAt(event->pos()) != 0)
	{
		mySelectionChanged();
	}	
}

void FeatureMapScene::onNewClicked()
{
	add(false);
}

void FeatureMapScene::onCopyClicked()
{
	add(true);
}

void FeatureMapScene::onDeleteSelectedClicked()
{
	erase();
}

void FeatureMapScene::onDeleteAllClicked()
{
	clear();
}

void FeatureMapScene::onDrawSourceChanged(QAction* action)
{
	FeatureMapItem* item = mySelectedItem();
	if (item != 0)
	{
		for (int index = DrawingSource_Begin; index < DrawingSource_End; index ++)
			if (m_actionDrawSource[index] == action)
		{
			mySelectedItem()->setSource((DrawingSource)index);
			updateSettings();
			return;
		}
	}
}

void FeatureMapScene::onDrawModeChanged(QAction* action)
{
	FeatureMapItem* item = mySelectedItem();
	if (item != 0)
	{
		for (int index = DrawingMode_Begin; index < DrawingMode_End; index ++)
			if (m_actionDrawMode[index] == action)
		{
			mySelectedItem()->setMode((DrawingMode)index);
			updateSettings();
			return;
		}
	}
}

void FeatureMapScene::onSaveItemClicked()
{
	FeatureMapItem* item = mySelectedItem();
	if (item != 0 && item->isVisible())
	{
		QString fileExt;
		QString fileName = QFileDialog::getSaveFileName(0, "Save canvas as image",
			"", "Image files (*.bmp *.png)", &fileExt);
		if (fileName.size() > 0)
		{
			const QGraphicsView* view = views().at(0);
			QPixmap image = QPixmap::grabWidget(
				view->viewport(),
				view->mapFromScene(item->mapToScene(item->boundingRect())).boundingRect());
			image.save(fileName, 0, 100);
		}
	}
}

void FeatureMapScene::onSaveSceneClicked()
{
	QString fileExt;
	QString fileName = QFileDialog::getSaveFileName(0, "Save canvas as image",
		"", "Image files (*.bmp *.png)", &fileExt);
	if (fileName.size() > 0)
	{
		const QGraphicsView* view = views().at(0);
		QPixmap image = QPixmap::grabWidget(
				view->viewport(), view->viewport()->rect());
		image.save(fileName, 0, 100);
	}
}

void FeatureMapScene::onSettingsItemClicked()
{
	FeatureMapItem* item = mySelectedItem();
	if (item != 0)
	{
		ItemSettingsDialog dlg(item->settings());
		if (dlg.exec() == QDialog::Accepted)
		{
			item->setSettings(dlg.settings());
			updateSettings();
		}
	}
}

void FeatureMapScene::onSettingsSceneClicked()
{
	SceneSettingsDialog dlg(settings());
	if (dlg.exec() == QDialog::Accepted)
	{
		setSettings(dlg.settings());
	}
}

void FeatureMapScene::add(bool asCopy)
{
	FeatureMapItem* item = new FeatureMapItem(m_settings,
		(asCopy == true && mySelectedItem() != 0) ? mySelectedItem()->settings() : ItemSettings());

	addItem(item);
	if (mySelectedItem() != 0)
	{
		item->setPos(mySelectedItem()->pos().operator +=(QPointF(36, 36)));
	}
	else
	{
		item->setPos(rand() % 200, rand() % 100);
	}
	updateSettings();
}

void FeatureMapScene::erase()
{
	foreach (QGraphicsItem* item, selectedItems())
	{
		removeItem(item);
	}
	updateSettings();
}

void FeatureMapScene::clear()
{
	foreach (QGraphicsItem* item, items())
	{
		removeItem(item);
	}
	updateSettings();
}

void FeatureMapScene::mySelectionChanged()
{
        // Always bring to front the selected item!
        if (selectedItems().empty() == true)
	{
		return;
	}

        QGraphicsItem* selectedItem = selectedItems().first();
        QList<QGraphicsItem*> overlapItems = selectedItem->collidingItems();

        qreal zValue = selectedItem->zValue();
        foreach (QGraphicsItem* item, overlapItems)
        {
                if (item->zValue() >= zValue)
			zValue = item->zValue() + 0.1;
        }
	if (zValue > selectedItem->zValue())
		selectedItem->setZValue(zValue);
	
	// Keep only one selected
	foreach (QGraphicsItem* item, items())
	{
		if (item != selectedItem)
			item->setSelected(false);
	}
}

QList<FeatureMapItem*> FeatureMapScene::myItems() const
{
	QList<FeatureMapItem*> mitems;
        foreach (QGraphicsItem* item, items())
        {
		mitems.append((FeatureMapItem*)item);	
	}
	return mitems;
}

FeatureMapItem* FeatureMapScene::mySelectedItem() const
{
	QList<QGraphicsItem*> sel_items = selectedItems();
	return sel_items.empty() ? 0 : (FeatureMapItem*)sel_items.at(0);
}

QRectF FeatureMapScene::visibleRect() const
{
	if (views().empty())
		return QRectF();
	const QGraphicsView* view = views()[0];
	return view->mapToScene(view->viewport()->geometry()).boundingRect();
}

void FeatureMapScene::updateSettings()
{
	m_settings.m_items.clear();
	foreach (QGraphicsItem* item, items())
	{
		m_settings.m_items.push_back(((FeatureMapItem*)item)->settings());
	}	
}
