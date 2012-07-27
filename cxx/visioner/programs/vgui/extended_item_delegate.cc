#include "extended_item_delegate.h"

void ExtendedItemDelegate::paint(QPainter* painter,
    const QStyleOptionViewItem& option, const QModelIndex& index) const
{
  QItemDelegate::paint(painter, option, index);
}
