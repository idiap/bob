#include "controls.h"
#include "visioner/util/util.h"

// Create a tool button using the icon file path
QToolButton* toolButton(const QString& iconPath, const QString& toolTip, const int size)
{
  QToolButton* button = new QToolButton();
  button->setIcon(QIcon(iconPath));
  button->setIconSize(QSize(size, size));
  button->setToolTip(toolTip);
  return button;
}

// Create a push button using the text label and the icon file path
QPushButton* pushButton(const QString& text, const QString& iconPath, const QString& toolTip)
{
  QPushButton* button = new QPushButton(QIcon(iconPath), text);
  button->setToolTip(toolTip);
  return button;
}

// Create a menu action
QAction* action(const QString& iconPath, const QString& text, bool checkable)
{
  QAction* action = new QAction(QIcon(iconPath), text, 0);
  action->setCheckable(checkable);
  return action;
}

// Create a separator for a dialog
QFrame* hSeparator()
{
  QFrame* frame = new QFrame(0);
  frame->setFrameStyle(QFrame::HLine | QFrame::Raised);
  return frame;
}

QFrame* vSeparator()
{
  QFrame* frame = new QFrame(0);
  frame->setFrameStyle(QFrame::VLine | QFrame::Raised);
  return frame;
}

// Updates a list of widgets
void updateBuddies(std::vector<QWidget*>& buddies)
{
  for (std::vector<QWidget*>::iterator itWidget = buddies.begin(); itWidget != buddies.end(); ++ itWidget)
  {
    QWidget* widget = *itWidget;
    if (widget != 0)
      widget->update();
  }
}

// Build the OK/Cancel layout
QHBoxLayout* buildOKCancelLayout(QDialog* parent)
{
  QPushButton* buttonOK = new QPushButton("OK");
  QPushButton* buttonCancel = new QPushButton("Cancel");

  QObject::connect(buttonOK, SIGNAL(clicked()), parent, SLOT(accept()));
  QObject::connect(buttonCancel, SIGNAL(clicked()), parent, SLOT(reject()));

  QHBoxLayout* hLayout = new QHBoxLayout();
  hLayout->addStretch();
  hLayout->addWidget(buttonOK);
  hLayout->addWidget(buttonCancel);
  return hLayout;
}

// Add a component to some horizontal layout
// NB: if some widget/layout is <0> it will be replaced by <addStretch>
void addToHLayout(QHBoxLayout* hLayout, QWidget* widget, int stretchFactor)
{
  if (widget == 0)
  {
    hLayout->addStretch(stretchFactor);
  }
  else
  {
    hLayout->addWidget(widget, stretchFactor);
  }
}
void addToHLayout(QHBoxLayout* hLayout, QLayout* layout, int stretchFactor)
{
  if (layout == 0)
  {
    hLayout->addStretch(stretchFactor);
  }
  else
  {
    hLayout->addLayout(layout, stretchFactor);
  }
}

// Build an icon used for displaying the current chosen color
QPixmap color_pixmap(const QColor& color, int w, int h)
{
  // Background color
  QPixmap pxm(w, h);
  pxm.fill(color);

  // 3D-sort-of border
  QPainter painter(&pxm);
  painter.setBrush(Qt::NoBrush);
  for (int i = 0; i < 4; i ++)
  {
    painter.setPen(QPen(color.lighter(140 - 10 * i), 1, Qt::SolidLine));
    painter.drawLine(i, i, w - i, i);
    painter.drawLine(i, i, i, h - i);
  }
  for (int i = 0; i < 4; i ++)
  {
    painter.setPen(QPen(color.darker(140 - 10 * i), 1, Qt::SolidLine));
    painter.drawLine(i, h - i, w, h - i);
    painter.drawLine(w - i, i, w - i, h);
  }

  return pxm;
}

// Catch paint event
void FontColorButton::paintEvent(QPaintEvent* event)
{
  QPushButton::paintEvent(event);

  QPainter painter(this);
  painter.setFont(m_font);
  painter.setPen(m_color);
  painter.setBrush(Qt::NoBrush);

  QFontMetrics fm = painter.fontMetrics();
  const int text_w = fm.width("font"), text_h = fm.height() * 2 / 3;
  const int text_x = (width() / 2 - text_w) / 2, text_y = (height() + text_h) / 2;
  painter.drawText(text_x, text_y, "font");

  const int dx = 2, dy = dx;
  const int icon_x = width() / 2 + dx, icon_y = dy;
  const int icon_w = width() - dx - icon_x, icon_h = height() - 2 * dy;
  painter.drawPixmap(icon_x, icon_y, color_pixmap(m_color, icon_w, icon_h));
}

// Catch the click event - try to change the current font
void FontColorButton::mousePressEvent(QMouseEvent* event)
{
  bool ok = false;
  if (event->x() < width() / 2)
  {
    m_font = QFontDialog::getFont(&ok, m_font);
  }
  else
  {
    m_color.setRgba(QColorDialog::getRgba(m_color.rgba()));
  }

  if (ok == true)
  {
    update();
    updateBuddies(m_buddies);
  }
}

// Catch paint event
void ColorButton::paintEvent(QPaintEvent* event)
{
  QPushButton::paintEvent(event);

  QPainter painter(this);
  const int dx = 2, dy = dx;
  painter.drawPixmap(dx, dy, width() - 2 * dx, height() - 2 * dy,
      color_pixmap(m_color, width() - 2 * dx, height() - 2 * dy));
}

// Catch the click event
void ColorButton::mousePressEvent(QMouseEvent* /*event*/)
{
  m_color.setRgba(QColorDialog::getRgba(m_color.rgba()));
  update();
  updateBuddies(m_buddies);
}

// Catch the checked change event
void ValueCheck::paintEvent(QPaintEvent* event)
{
  QCheckBox::paintEvent(event);

  if (m_painted == false)
  {
    m_painted = true;
    setChecked(m_value);
  }

  else if (m_value != (checkState() == Qt::Checked))
  {
    m_value = checkState() == Qt::Checked;
  }
}

// Catch the checked change event
void ExtendedRadioButton::paintEvent(QPaintEvent* event)
{
  QRadioButton::paintEvent(event);

  if (m_painted == false || isChecked() != m_checked)
  {
    m_painted = true;
    m_checked = isChecked();
  }
}

// Constructor
  RenameTabBar::RenameTabBar(QWidget* parent)
: QTabBar(parent)
{
}

// Catch the double click event to handle renaming
void RenameTabBar::mouseDoubleClickEvent(QMouseEvent* event)
{
  const int tabCount = count();
  for (int i = 0; i < tabCount; i ++)
  {
    if (tabRect(i).contains(event->pos()))
    {
      // Show the input dialog with the old name and check if renaming occured
      bool ok;
      QString text = QInputDialog::getText(
          this,
          "Rename view",
          "Enter new name for the selected view:",
          QLineEdit::Normal,
          tabText(i), &ok);
      if (ok && !text.isEmpty())
      {
        setTabText(i, text);
      }
      return;
    }
  }
}

// Constructor
  RenameTabWidget::RenameTabWidget(QWidget* parent)
: QTabWidget(parent)
{
  setTabBar(new RenameTabBar());
}

// Constructor
  GradientLabel::GradientLabel(const QColor& colorMin, const QColor& colorMax, QWidget* parent)
:	QLabel(parent),
  m_colorMin(colorMin), m_colorMax(colorMax)
{
}

// Catch the paint event - draw the gradient
void GradientLabel::paintEvent(QPaintEvent* /*event*/)
{
  // Draw the frame
  QPainter painter(this);
  painter.setRenderHint(QPainter::Antialiasing);
  painter.setRenderHint(QPainter::TextAntialiasing);
  painter.setRenderHint(QPainter::SmoothPixmapTransform);
  QLabel::drawFrame(&painter);

  // Draw the gradient
  QLinearGradient gradient(0, 0, 1, 0);
  gradient.setCoordinateMode(QGradient::ObjectBoundingMode);
  gradient.setColorAt(0.0, m_colorMin);
  gradient.setColorAt(1.0, m_colorMax);
  painter.fillRect(0, 0, width(), height(), gradient);
}
