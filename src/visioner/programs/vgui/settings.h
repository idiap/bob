/**
 * @file visioner/programs/vgui/settings.h
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

#ifndef SETTINGS_H
#define SETTINGS_H

#include <QRectF>
#include <QString>
#include <QColor>
#include <QFont>
#include <QImage>
#include <QBrush>
#include <QPainter>
#include <vector>

#if (defined(__LP64__) || defined(__APPLE__))
// Serialization
inline QDataStream& operator<<(QDataStream& stream, const std::size_t& settings)
{
  stream << (quint32)settings;
  return stream;
}
inline QDataStream& operator>>(QDataStream& stream, std::size_t& settings)
{
  quint32 value;
  stream >> value;
  settings = (std::size_t)value;
  return stream;
}
#endif

// Serialization
inline QDataStream& operator<<(QDataStream& stream, const std::string& settings)
{
  stream << QString(settings.c_str());
  return stream;
}
inline QDataStream& operator>>(QDataStream& stream, std::string& settings)
{
  QString value;
  stream >> value;
  settings = value.toStdString();
  return stream;
}

// Serialization
  template <typename T>
QDataStream& operator<<(QDataStream& stream, const std::vector<T>& settings)
{
  stream << (int)settings.size();
  for (std::size_t i = 0; i < settings.size(); i ++)
  {
    stream << settings[i];
  }

  return stream;
}

  template <typename T>
QDataStream& operator>>(QDataStream& stream, std::vector<T>& settings)
{
  int size;
  stream >> size;
  settings.resize(size);
  for (std::size_t i = 0; i < settings.size(); i ++)
  {
    stream >> settings[i];
  }

  return stream;
}

// Background drawing type
enum Background
{
  Background_Begin = 0,

  Background_Fill = Background_Begin,
  Background_Pattern,

  Background_End
};
const QString BackgroundStr[Background_End] =
{
  "Fill",
  "Pattern"
};

// Serialization
inline QDataStream& operator<<(QDataStream& stream, const Background& settings)
{
  stream << (int)settings;
  return stream;
}
inline QDataStream& operator>>(QDataStream& stream, Background& settings)
{
  int value;
  stream >> value;
  settings = (Background)value;
  return stream;
}

// Positions for displaying text infos
enum TextInfoPos
{
  TextInfoPos_Begin = 0,

  TextInfoPos_TopLeft = TextInfoPos_Begin,
  TextInfoPos_TopRight,
  TextInfoPos_BottomLeft,
  TextInfoPos_BottomRight,

  TextInfoPos_End
};
const QString TextInfoPosStr[TextInfoPos_End] =
{
  "Top-left",
  "Top-right",
  "Bottom-left",
  "Bottom-right"
};

// Serialization
inline QDataStream& operator<<(QDataStream& stream, const TextInfoPos& settings)
{
  stream << (int)settings;
  return stream;
}
inline QDataStream& operator>>(QDataStream& stream, TextInfoPos& settings)
{
  int value;
  stream >> value;
  settings = (TextInfoPos)value;
  return stream;
}

// Drawing modes
enum DrawingSource
{
  DrawingSource_Begin = 0,

  DrawingSource_Greys = DrawingSource_Begin,	// Greyscale values

  DrawingSource_LBP,				// MB-LBP map        
  DrawingSource_mLBP,				// MB-mLBP map        
  DrawingSource_tLBP,				// MB-tLBP map
  DrawingSource_dLBP,				// MB-dLBP map

  DrawingSource_End
};
const QString DrawingSourceStr[DrawingSource_End] =
{
  "Grey",

  "LBP",          
  "mLBP", 
  "tLBP",
  "dLBP"
};

// Serialization
inline QDataStream& operator<<(QDataStream& stream, const DrawingSource& settings)
{
  stream << (int)settings;
  return stream;
}
inline QDataStream& operator>>(QDataStream& stream, DrawingSource& settings)
{
  int value;
  stream >> value;
  settings = (DrawingSource)value;
  return stream;
}

enum DrawingMode
{
  DrawingMode_Begin = 0,

  DrawingMode_Original = DrawingMode_Begin,		// Original data
  DrawingMode_Histogram,					// Histogram
  DrawingMode_CumHistogram,				// Cumulative histogram

  DrawingMode_End
};
const QString DrawingModeStr[DrawingMode_End] =
{
  "Original",
  "Histogram",
  "Cumulative histogram"
};

// Serialization
inline QDataStream& operator<<(QDataStream& stream, const DrawingMode& settings)
{
  stream << (int)settings;
  return stream;
}
inline QDataStream& operator>>(QDataStream& stream, DrawingMode& settings)
{
  int value;
  stream >> value;
  settings = (DrawingMode)value;
  return stream;
}

// Drawing settings for a frame/Item: background + border
struct FrameSettings
{
  // Constructor
  FrameSettings(	const QColor& backColor, const QColor& backPatternColor,
      Background backType,
      const QColor& borderColor, int borderSize,
      const QColor& borderResizeColor, bool borderResizeEnable)
    :
      m_backColor(backColor),
      m_backPatternColor(backPatternColor),
      m_backType(backType),
      m_borderColor(borderColor),
      m_borderSize(borderSize),
      m_borderResizeColor(borderResizeColor),
      m_borderResizeEnable(borderResizeEnable),
      m_pattImage(16, 16, QImage::Format_RGB32),
      m_pattBrush(Qt::SolidPattern),

      m_oldBackColor(0, 0, 0, 0),
      m_oldBackPatternColor(0, 0, 0, 0)
      {
      }

  // Draw this frame on some painting device
  void drawBackground(QPainter& painter, const QRectF& drawRect) const;
  void drawBackground(QPainter& painter, const QRectF& drawRect, const std::vector<QRectF>& excludeRects) const;
  void drawBackground(QPainter& painter, const QRectF& drawRect, const QRectF& excludeRect) const;
  void drawBorder(QPainter& painter, const QRectF& drawRect) const;

  // Get the border resize handlers' dimensions
  static int resizeHandlerWidth() { return 8; }
  static int resizeHandlerHeight() { return 8; }

  // Check if some point is within some resizing area
  static bool topLeftResizable(const QRectF& drawRect, const QPointF& mousePoint);
  static bool topRightResizable(const QRectF& drawRect, const QPointF& mousePoint);
  static bool bottomLeftResizable(const QRectF& drawRect, const QPointF& mousePoint);
  static bool bottomRightResizable(const QRectF& drawRect, const QPointF& mousePoint);

  // Attributes
  QColor			m_backColor;
  QColor			m_backPatternColor;
  Background		m_backType;

  QColor			m_borderColor;		// General border color
  int			m_borderSize;
  QColor			m_borderResizeColor;	// Resize handlers on the border
  bool			m_borderResizeEnable;	// Resize handlers flag

  mutable QImage		m_pattImage;		// Texture for the background pattern
  mutable QBrush		m_pattBrush;		// Brush for the background pattern

  private:

  // To keep track of the background pattern's modifications
  mutable QColor		m_oldBackColor;
  mutable QColor		m_oldBackPatternColor;
};

// Serialization
QDataStream& operator<<(QDataStream& stream, const FrameSettings& settings);
QDataStream& operator>>(QDataStream& stream, FrameSettings& settings);

// Frame displaying text information on multiple lines
struct TextInfoSettings
{
  // Text displayed in one line
  struct Line
  {
    QString		m_text, m_description;
    QFont		m_font;
    QColor		m_color;
    bool		m_enable;
  };
  typedef std::vector<Line>::iterator		ItLine;
  typedef std::vector<Line>::const_iterator	ConstItLine;

  // Constructor
  TextInfoSettings();

  // Initialize line with the default settings
  void init();

  // Update the text information at some specified position
  bool update(std::size_t index, const QString& text)
  {
    if (index < m_lines.size())
    {
      m_lines[index].m_text = text;
      return true;
    }
    return false;
  }

  // Check if any line is enabled
  bool anyLineEnabled() const;

  // Attributes
  bool			m_enable;
  TextInfoPos		m_position;
  FrameSettings		m_frame;
  std::vector<Line>	m_lines;
};

// Serialization
QDataStream& operator<<(QDataStream& stream, const TextInfoSettings::Line& settings);
QDataStream& operator>>(QDataStream& stream, TextInfoSettings::Line& settings);

// Serialization
QDataStream& operator<<(QDataStream& stream, const TextInfoSettings& settings);
QDataStream& operator>>(QDataStream& stream, TextInfoSettings& settings);

// Drawing settings for an item
struct ItemSettings
{
  // Constructor
  ItemSettings()
    : 	m_source(DrawingSource_Greys),
    m_mode(DrawingMode_Original),

    m_cx(1), m_cy(1),

    m_gt_bbx(false),
    m_gtBbxColor(0, 225, 0, 255),

    m_gt_keypoints(false),
    m_gt_keyLabel(false),
    m_gtKeypointsFont("Sans", 8, QFont::Bold),
    m_gtKeypointsFontColor(225, 0, 25, 255),
    m_gtKeypointsColor(0, 0, 225, 255),

    m_gt_label(false),
    m_gtLabelFont("Sans", 12, QFont::Bold),
    m_gtLabelFontColor(0, 25, 225, 255),
    m_gtLabelColor(255, 255, 127, 255),

    m_histoBinColor(55, 55, 65, 255),
    m_histoOutlineColor(0, 0, 0, 255),
    m_histoTextFont("Sans", 8, QFont::Bold),
    m_histoTextColor(0, 0, 175, 255),

    m_textInfo(),

    m_scene_pos(rand() % 640, rand() % 480),
    m_scene_size(320, 240)
    {
      m_gtKeypointsFont.setUnderline(true);
    }

  // Multi-block cell range
  static std::size_t min_cell_size() { return 1; }
  static std::size_t max_cell_size() { return 8; }

  // Increase/Decrease the cell size
  bool inc_cx() { return inc(m_cx); }
  bool inc_cy() { return inc(m_cy); }

  bool dec_cx() { return dec(m_cx); }
  bool dec_cy() { return dec(m_cy); }

  static bool inc(std::size_t& csize)
  {
    if (csize >= max_cell_size())
    {
      return false;
    }
    csize ++;
    return true;
  }
  static bool dec(std::size_t& csize)
  {
    if (csize <= min_cell_size())
    {
      return false;
    }
    csize --;
    return true;
  }

  // Equality operator
  bool operator==(const ItemSettings& other) const
  {
    return	m_source == other.m_source &&
      m_mode == other.m_mode &&
      m_cx == other.m_cx &&
      m_cy == other.m_cy;
  }
  bool operator!=(const ItemSettings& other) const
  {
    return !(*this == other);
  }

  // Attributes	
  DrawingSource		m_source;          	// Data type to draw
  DrawingMode		m_mode;			// Drawing type

  std::size_t		m_cx, m_cy;		// Cell size of the multi-block features

  bool			m_gt_bbx;		// Ground truth bbx
  QColor			m_gtBbxColor;

  bool			m_gt_keypoints;		// Ground truth keypoints
  bool                    m_gt_keyLabel;          // Ground truth keypoints label
  QFont			m_gtKeypointsFont;
  QColor			m_gtKeypointsFontColor;
  QColor			m_gtKeypointsColor;

  bool			m_gt_label;		// Ground truth label
  QFont			m_gtLabelFont;
  QColor			m_gtLabelFontColor;
  QColor			m_gtLabelColor;

  QColor			m_histoBinColor;	// Histogram
  QColor			m_histoOutlineColor;	
  QFont			m_histoTextFont;
  QColor			m_histoTextColor;

  TextInfoSettings	m_textInfo;		// Text information to display

  QPointF			m_scene_pos;		// Position and size
  QSizeF			m_scene_size;		//	in the scene coordinates
};

// Serialization
QDataStream& operator<<(QDataStream& stream, const ItemSettings& settings);
QDataStream& operator>>(QDataStream& stream, ItemSettings& settings);

// Drawing settings for the whole scene
struct SceneSettings
{
  // Constructor
  SceneSettings(int n_items = 1)
    : 	// Container frame - no border there
      m_frame(QColor(255, 255, 255, 255), QColor(225, 225, 255, 255),
          Background_Pattern,
          QColor(185, 185, 185, 255), 2,
          QColor(95, 95, 155, 255), false),

      // Selected Item' frame - larger border
      m_selFrame(QColor(255, 255, 255, 255), QColor(225, 225, 255, 255),
          Background_Fill,
          QColor(245, 245, 125, 255), 4,
          QColor(95, 95, 155, 255), true),

      // Not selected Item' frame - not that larger border
      m_notSelFrame(QColor(255, 255, 255, 255), QColor(225, 225, 255, 255),
          Background_Fill,
          QColor(125, 125, 245, 255), 2,
          QColor(95, 95, 155, 255), false),

      // items
      m_items(n_items)
      {
      }

  // Attributes
  FrameSettings			m_frame;		// Frame settings for the whole container
  FrameSettings			m_selFrame;		// Frame settings for the selected items
  FrameSettings			m_notSelFrame;		// Frame settings for the unselected items
  std::vector<ItemSettings>	m_items;		// Settings for each item		
};

// Serialization
QDataStream& operator<<(QDataStream& stream, const SceneSettings& settings);
QDataStream& operator>>(QDataStream& stream, SceneSettings& settings);

#endif // SETTINGS_H
