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
