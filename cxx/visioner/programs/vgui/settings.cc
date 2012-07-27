#include "settings.h"

void FrameSettings::drawBackground(QPainter& painter, const QRectF& drawRect) const
{
  switch (m_backType)
  {
    case Background_Fill:		// Simple fill
      painter.fillRect(drawRect, QBrush(m_backColor, Qt::SolidPattern));
      break;

    case Background_Pattern:	// Rectangular pattern
      // Update the pattern brush if needed
      if (m_backPatternColor != m_oldBackPatternColor || m_backColor != m_oldBackColor)
      {
        static const int dfill = 8, dfill2 = 2 * dfill;
        for (int i = 0; i < dfill2; i ++)
        {
          const QRgb color1 = i < dfill ? m_backPatternColor.rgb() : m_backColor.rgb();
          for (int j = 0; j < dfill; j ++)
          {
            m_pattImage.setPixel(i, j, color1);
          }

          const QRgb color2 = i < dfill ? m_backColor.rgb() : m_backPatternColor.rgb();
          for (int j = dfill; j < dfill2; j ++)
          {
            m_pattImage.setPixel(i, j, color2);
          }
        }

        m_pattBrush.setTextureImage(m_pattImage);
      }
      m_oldBackColor = m_backColor;
      m_oldBackPatternColor = m_backPatternColor;

      // Just fill the drawing area with the pattern brush
      painter.fillRect(drawRect, m_pattBrush);
      break;

    default:
      break;
  }
}

void FrameSettings::drawBackground(QPainter& painter, const QRectF& drawRect,
    const std::vector<QRectF>&) const
{
  // Build the region that is to be clipped (big rectangle - excluding rectangles)
  //QPainterPath pathAll;
  //pathAll.addRect(drawRect);

  //QPainterPath pathExclude;
  //for (std::vector<QRectF>::const_iterator it = excludeRects.begin(); it != excludeRects.end(); ++ it)
  //{
  //	pathExclude.addRect(*it);
  //}

  // Set the clipping region and draw the background
  //painter.setClipping(true);
  //painter.setClipPath(pathAll.subtracted(pathExclude));
  drawBackground(painter, drawRect);
  //painter.setClipping(false);
}

void FrameSettings::drawBackground(QPainter& painter, const QRectF& drawRect, const QRectF& excludeRect) const
{
  // Build the region that is to be clipped (big rectangle - excluding rectangle)
  QPainterPath pathAll;
  pathAll.addRect(drawRect);

  QPainterPath pathExclude;
  pathExclude.addRect(excludeRect);

  // Set the clipping region and draw the background
  //painter.setClipping(true);
  //painter.setClipPath(pathAll.subtracted(pathExclude));
  drawBackground(painter, drawRect);
  //painter.setClipping(false);
}

void FrameSettings::drawBorder(QPainter& painter, const QRectF& drawRect) const
{
  const int dx = resizeHandlerWidth(), dy = resizeHandlerHeight();
  const int left = drawRect.left(), right = drawRect.right();
  const int top = drawRect.top(), bottom = drawRect.bottom();

  // Draw the border
  painter.setBrush(Qt::NoBrush);
  painter.setPen(QPen(m_borderColor, m_borderSize, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
  painter.drawRect(left + m_borderSize / 2, top + m_borderSize / 2,
      drawRect.width() - m_borderSize, drawRect.height() - m_borderSize);

  // Draw the resize areas of the border
  if (m_borderResizeEnable == true)
  {
    painter.setPen(QPen(m_borderResizeColor, m_borderSize, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));

    // ... resize: top-left
    painter.drawLine(left, top, left + dx, top);
    painter.drawLine(left, top, left, top + dy);

    // ... resize: top-right
    painter.drawLine(right - dx, top, right, top);
    painter.drawLine(right, top, right, top + dy);

    // ... resize: bottom-left
    painter.drawLine(left, bottom, left + dx, bottom);
    painter.drawLine(left, bottom, left, bottom - dy);

    // ... resize: bottom-right
    painter.drawLine(right - dx, bottom, right, bottom);
    painter.drawLine(right, bottom, right, bottom - dy);
  }
}

bool FrameSettings::topLeftResizable(const QRectF& drawRect, const QPointF& mousePoint)
{
  return (QRectF(	drawRect.left(), drawRect.top(),
        resizeHandlerWidth(), resizeHandlerHeight()))
    .contains(mousePoint);
}

bool FrameSettings::topRightResizable(const QRectF& drawRect, const QPointF& mousePoint)
{
  return (QRectF(	drawRect.right() - resizeHandlerWidth(), drawRect.top(),
        resizeHandlerWidth(), resizeHandlerHeight()))
    .contains(mousePoint);
}

bool FrameSettings::bottomLeftResizable(const QRectF& drawRect, const QPointF& mousePoint)
{
  return (QRectF(	drawRect.left(), drawRect.bottom() - resizeHandlerHeight(),
        resizeHandlerWidth(), resizeHandlerHeight()))
    .contains(mousePoint);
}

bool FrameSettings::bottomRightResizable(const QRectF& drawRect, const QPointF& mousePoint)
{
  return (QRectF(	drawRect.right() - resizeHandlerWidth(), drawRect.bottom() - resizeHandlerHeight(),
        resizeHandlerWidth(), resizeHandlerHeight()))
    .contains(mousePoint);
}

QDataStream& operator<<(QDataStream& stream, const FrameSettings& settings)
{
  stream << settings.m_backColor;
  stream << settings.m_backPatternColor;
  stream << settings.m_backType;	  
  stream << settings.m_borderColor;	
  stream << settings.m_borderSize;
  stream << settings.m_borderResizeColor;	
  stream << settings.m_borderResizeEnable;          

  return stream;
}

QDataStream& operator>>(QDataStream& stream, FrameSettings& settings)
{
  stream >> settings.m_backColor;
  stream >> settings.m_backPatternColor;
  stream >> settings.m_backType;	  
  stream >> settings.m_borderColor;	
  stream >> settings.m_borderSize;
  stream >> settings.m_borderResizeColor;	
  stream >> settings.m_borderResizeEnable;

  return stream;
}

  TextInfoSettings::TextInfoSettings()
:	m_enable(true),
  m_position(TextInfoPos_BottomRight),
  m_frame(QColor(225, 225, 255, 175), QColor(125, 125, 125, 255),
      Background_Fill, 
      QColor(55, 55, 55, 255), 1,
      QColor(95, 95, 155, 255), false),
  m_lines()
{
  init();
}

bool TextInfoSettings::anyLineEnabled() const
{
  for (ConstItLine it = m_lines.begin(); it != m_lines.end(); ++ it)
    if (it->m_enable == true)
    {
      return true;
    }
  return false;
}

void TextInfoSettings::init()
{
  m_lines.clear();

  // Drawing source
  {
    TextInfoSettings::Line line;
    line.m_text = "Pixels::Original";
    line.m_description = "Drawing source.";
    line.m_font = QFont("Sans", 8, QFont::Normal);
    line.m_color = QColor(25, 25, 155, 255);
    line.m_enable = true;
    m_lines.push_back(line);
  }

  // Source image name
  {
    TextInfoSettings::Line line;
    line.m_text = "Source name";
    line.m_description = "Source name.";
    line.m_font = QFont("Sans", 8, QFont::Normal);
    line.m_color = QColor(0, 0, 0, 255);
    line.m_enable = true;
    m_lines.push_back(line);
  }

  // Drawing time
  {
    TextInfoSettings::Line line;
    line.m_text = "Drawing time [ms]";
    line.m_description = "Item drawing time.";
    line.m_font = QFont("Sans", 8, QFont::Bold);
    line.m_color = QColor(225, 0, 0, 255);
    line.m_enable = false;
    m_lines.push_back(line);
  }
}

QDataStream& operator<<(QDataStream& stream, const TextInfoSettings::Line& settings)
{
  stream << settings.m_color;
  stream << settings.m_description;
  stream << settings.m_enable;
  stream << settings.m_font;
  stream << settings.m_text;

  return stream;
}

QDataStream& operator>>(QDataStream& stream, TextInfoSettings::Line& settings)
{
  stream >> settings.m_color;
  stream >> settings.m_description;
  stream >> settings.m_enable;
  stream >> settings.m_font;
  stream >> settings.m_text;

  return stream;
}

QDataStream& operator<<(QDataStream& stream, const TextInfoSettings& settings)
{
  stream << settings.m_enable;
  stream << settings.m_position;
  stream << settings.m_frame;
  stream << settings.m_lines;

  return stream;
}

QDataStream& operator>>(QDataStream& stream, TextInfoSettings& settings)
{
  stream >> settings.m_enable;
  stream >> settings.m_position;
  stream >> settings.m_frame;
  stream >> settings.m_lines;

  return stream;
}

QDataStream& operator<<(QDataStream& stream, const ItemSettings& settings)
{
  stream << settings.m_source;
  stream << settings.m_mode;	
  stream << settings.m_cx;
  stream << settings.m_cy;	
  stream << settings.m_gt_bbx;		
  stream << settings.m_gtBbxColor;
  stream << settings.m_gt_keypoints;		
  stream << settings.m_gtKeypointsFont;
  stream << settings.m_gtKeypointsFontColor;
  stream << settings.m_gtKeypointsColor;
  stream << settings.m_gt_label;		
  stream << settings.m_gtLabelFont;
  stream << settings.m_gtLabelFontColor;
  stream << settings.m_gtLabelColor;
  stream << settings.m_histoBinColor;	
  stream << settings.m_histoOutlineColor;	
  stream << settings.m_histoTextFont;
  stream << settings.m_histoTextColor;
  stream << settings.m_textInfo;		
  stream << settings.m_scene_pos;
  stream << settings.m_scene_size;

  return stream;
}

QDataStream& operator>>(QDataStream& stream, ItemSettings& settings)
{
  stream >> settings.m_source;
  stream >> settings.m_mode;	
  stream >> settings.m_cx;
  stream >> settings.m_cy;	
  stream >> settings.m_gt_bbx;		
  stream >> settings.m_gtBbxColor;
  stream >> settings.m_gt_keypoints;		
  stream >> settings.m_gtKeypointsFont;
  stream >> settings.m_gtKeypointsFontColor;
  stream >> settings.m_gtKeypointsColor;
  stream >> settings.m_gt_label;		
  stream >> settings.m_gtLabelFont;
  stream >> settings.m_gtLabelFontColor;
  stream >> settings.m_gtLabelColor;
  stream >> settings.m_histoBinColor;	
  stream >> settings.m_histoOutlineColor;	
  stream >> settings.m_histoTextFont;
  stream >> settings.m_histoTextColor;
  stream >> settings.m_textInfo;
  stream >> settings.m_scene_pos;
  stream >> settings.m_scene_size;

  return stream;
}

QDataStream& operator<<(QDataStream& stream, const SceneSettings& settings)
{
  stream << settings.m_frame;
  stream << settings.m_selFrame;
  stream << settings.m_notSelFrame;
  stream << settings.m_items;

  return stream;	
}

QDataStream& operator>>(QDataStream& stream, SceneSettings& settings)
{
  stream >> settings.m_frame;
  stream >> settings.m_selFrame;
  stream >> settings.m_notSelFrame;
  stream >> settings.m_items;

  return stream;
}
