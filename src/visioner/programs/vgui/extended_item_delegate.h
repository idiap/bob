/**
 * @file visioner/programs/vgui/extended_item_delegate.h
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#ifndef EXTENDED_ITEM_DELEGATE_H 
#define EXTENDED_ITEM_DELEGATE_H

#include <QtGui>

/**
 * Qt item delegate to be used within an ExtendedTable object. Mainly because
 * there might be elements in a table that need to be drawn different.
 */

class ExtendedItemDelegate : public QItemDelegate {

  Q_OBJECT

  public:

    // Constructor
    ExtendedItemDelegate(QWidget* parent)
      :	QItemDelegate(parent)
    {
    }

    // Overriden - paint the cell accordingly with the stored data type
    void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const;

};

#endif
