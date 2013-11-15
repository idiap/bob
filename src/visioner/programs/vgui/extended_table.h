/**
 * @file visioner/programs/vgui/extended_table.h
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

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
