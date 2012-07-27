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
