/**
 * @file visioner/programs/vgui/extended_table.cc
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

#include "extended_table.h"
#include "extended_item_delegate.h"

ExtendedTable::ExtendedTable(const QStringList& colNames, const QList<int>& colWidths)
	: 	QTableWidget(0, colNames.size(), 0)
{
	// Set header
	setHorizontalHeaderLabels(colNames);
	const int noCols = (int)colWidths.size();
	for (int i = 0; i < noCols; i ++)
		setColumnWidth(i, colWidths[i]);

	// Set general behaviour
	setAlternatingRowColors(true);
	setSelectionMode(QAbstractItemView::ContiguousSelection);
	setSelectionBehavior(QAbstractItemView::SelectRows);
	setEditTriggers(QAbstractItemView::NoEditTriggers | QAbstractItemView::SelectedClicked);
	setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
	horizontalHeader()->setStretchLastSection(true);
	
	setItemDelegate(new ExtendedItemDelegate(this));

	// Drag and drop support
	setDragEnabled(true);
	setAcceptDrops(false);
	setDropIndicatorShown(false);
	setDragDropMode(QAbstractItemView::NoDragDrop);
	
	// Disable the vertical header - resizeToContent is used!
	verticalHeader()->setDisabled(true);
}

void ExtendedTable::dataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight)
{
	QTableWidget::dataChanged(topLeft, bottomRight);
	resizeRowsToContents();
}
