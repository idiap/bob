#ifndef EXTENDED_TABLE_H
#define EXTENDED_TABLE_H

#include <QtGui>

/**
 * Qt table with some extensions.
 */

class ExtendedTable : public QTableWidget
{
	Q_OBJECT

public:

	// Constructor
	ExtendedTable(const QStringList& colNames, const QList<int>& colWidths);
	
protected:
	
	// Overriden event
	void dataChanged(const QModelIndex& topLeft, const QModelIndex& bottomRight);		
};

#endif
