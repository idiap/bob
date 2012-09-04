/**
 * @file visioner/programs/vgui/settings_dialog.cc
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

#include "bob/visioner/model/param.h"

#include "settings_dialog.h"
#include "controls.h"
#include "extended_table.h"

namespace impl {
  // Custom frame (background + border) label
  class FrameLabel : public QLabel
  {
    public:

      // Constructor
      FrameLabel(const FrameSettings& frame, QWidget* parent = 0)
        : QLabel(parent), m_frame(frame)
      {
      }

    protected:

      // Catch the paint event
      void paintEvent(QPaintEvent* /*event*/)
      {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);
        painter.setRenderHint(QPainter::TextAntialiasing);
        painter.setRenderHint(QPainter::SmoothPixmapTransform);

        m_frame.drawBackground(painter, painter.window());
        m_frame.drawBorder(painter, painter.window());
        //QLabel::drawFrame(&painter);
      }

    private:

      // Attributes
      const FrameSettings&	m_frame;
  };

  // Build a group box from frame settings
  QGroupBox* buildLayout(FrameSettings& frame, const QString& title)
  {
    // Background color
    ColorButton* buttonBackColor = new ColorButton(frame.m_backColor);

    // Background pattern color
    ColorButton* buttonBackPatternColor = new ColorButton(frame.m_backPatternColor);

    // Background type
    ValueCombo<Background>* comboBackType = new ValueCombo<Background>(frame.m_backType);
    for (int i = Background_Begin; i < Background_End; i ++)
    {
      comboBackType->add(BackgroundStr[i], (Background)i);
    }

    // Border color
    ColorButton* buttonBorderColor = new ColorButton(frame.m_borderColor);

    // Border resize color
    ColorButton* buttonBorderResizeColor = new ColorButton(frame.m_borderResizeColor);
    buttonBorderResizeColor->setEnabled(frame.m_borderResizeEnable);

    // Border size
    ValueCombo<int>* comboBorderSize = new ValueCombo<int>(frame.m_borderSize);
    comboBorderSize->add("Border x1", 1);
    comboBorderSize->add("Border x2", 2);
    comboBorderSize->add("Border x3", 3);
    comboBorderSize->add("Border x4", 4);
    comboBorderSize->add("Border x5", 5);
    comboBorderSize->add("Border x6", 6);

    // Frame preview label
    FrameLabel* labelFrame = new FrameLabel(frame);
    labelFrame->setFixedHeight(80);

    buttonBackColor->addUpdateBuddy(labelFrame);
    buttonBackPatternColor->addUpdateBuddy(labelFrame);
    comboBackType->addUpdateBuddy(labelFrame);
    buttonBorderColor->addUpdateBuddy(labelFrame);
    buttonBorderResizeColor->addUpdateBuddy(labelFrame);
    comboBorderSize->addUpdateBuddy(labelFrame);

    // Build the groupbox
    QGroupBox* group = new QGroupBox(title);
    QGridLayout* layout = new QGridLayout();
    int row = 0;
    layout->addWidget(new QLabel("Background type:"), row, 0, 1, 2);
    layout->addWidget(comboBackType, row, 2, 1, 4);
    row ++;
    layout->addWidget(new QLabel("Background colors:"), row, 0, 1, 2);
    layout->addWidget(buttonBackColor, row, 2, 1, 2);
    layout->addWidget(buttonBackPatternColor, row, 4, 1, 2);
    row ++;
    layout->addWidget(new QLabel("Border & handler:"), row, 0, 1, 2);
    layout->addWidget(buttonBorderColor, row, 2, 1, 2);
    layout->addWidget(buttonBorderResizeColor, row, 4, 1, 2);
    row ++;
    layout->addWidget(new QLabel("Border & handler size:"), row, 0, 1, 2);
    layout->addWidget(comboBorderSize, row, 2, 1, 4);
    row ++;
    layout->addWidget(new QLabel("Preview:"), row, 0, 4, 2);
    layout->addWidget(labelFrame, row, 2, 4, 4);

    group->setLayout(layout);

    return group;
  }

  // Build a group box from text information settings
  QGroupBox* buildLayout(TextInfoSettings& ti, const QString& title)
  {
    // Enable
    ValueCheck* checkTiEnable = new ValueCheck(ti.m_enable, "Show text information");

    // Build the table with the line settings			
    QStringList colNames;
    colNames << "  " << "Font & Color" << "Description";

    QList<int> colWidths;
    colWidths << 30 << 140 << 100;
    ExtendedTable* tableTiLines = new ExtendedTable(colNames, colWidths);

    tableTiLines->setRowCount(ti.m_lines.size());
    for (std::size_t i = 0; i < ti.m_lines.size(); i ++)
    {
      TextInfoSettings::Line& line = ti.m_lines[i];

      ValueCheck* checkTiLineEnable = new ValueCheck(line.m_enable, "");
      FontColorButton* buttonTiLineFont = new FontColorButton(line.m_font, line.m_color);

      tableTiLines->setCellWidget(i, 0, checkTiLineEnable);
      tableTiLines->setCellWidget(i, 1, buttonTiLineFont);
      tableTiLines->setItem(i, 2, new QTableWidgetItem(line.m_description));
    }

    // Build the groupbox
    QGroupBox* group = new QGroupBox(title);
    QVBoxLayout* vLayout = new QVBoxLayout();
    {
      vLayout->addWidget(checkTiEnable);
      vLayout->addWidget(hSeparator());
      vLayout->addWidget(tableTiLines);
    }
    group->setLayout(vLayout);

    return group;
  }

  // Build a group box from text information frame settings
  QGroupBox* buildGroupBox(TextInfoSettings& ti, const QString& title)
  {
    // Text frame position
    ValueCombo<TextInfoPos>* comboTiPos = new ValueCombo<TextInfoPos>(ti.m_position);
    for (int i = TextInfoPos_Begin; i < TextInfoPos_End; i ++)
      comboTiPos->add(TextInfoPosStr[i], (TextInfoPos)i);

    // Build the groupbox
    QGroupBox* group = new QGroupBox(title);
    QVBoxLayout* vLayout = new QVBoxLayout();
    {
      vLayout->addLayout(buildHLayout(new QLabel("Position:"), comboTiPos));
      vLayout->addWidget(buildLayout(ti.m_frame, "Frame"));
    }
    group->setLayout(vLayout);

    return group;
  }

  // Build a tab with the specified widget
  QWidget* buildTab(QWidget* content)
  {
    QVBoxLayout* vLayout = new QVBoxLayout();
    vLayout->addWidget(content);
    vLayout->addStretch();

    QWidget* widget = new QWidget();
    widget->setLayout(vLayout);

    return widget;
  }
}

void SceneSettingsDialog::buildGUI()
{
  QTabWidget* tabs = new QTabWidget();
  tabs->setTabPosition(QTabWidget::North);

  // Global frame settings
  tabs->addTab(	impl::buildTab(
        impl::buildLayout(m_settings.m_frame, "Frame")),
      "Global");

  // Selected item settings
  tabs->addTab(	impl::buildTab(
        impl::buildLayout(m_settings.m_selFrame, "Frame")),
      "Selected item");

  // Not selected item settings
  tabs->addTab(	impl::buildTab(
        impl::buildLayout(m_settings.m_notSelFrame, "Frame")),
      "Unselected item");

  // Build the main layout
  QVBoxLayout* mainLayout = new QVBoxLayout();
  mainLayout->addWidget(tabs);
  mainLayout->addLayout(buildOKCancelLayout(this));

  setLayout(mainLayout);
  setWindowTitle("Container Drawing Settings");
  //setFixedSize(QSize(480, 400));
  resize(QSize(480, 400));
  setModal(true);
}

void ItemSettingsDialog::buildGUI()
{
  QTabWidget* tabs = new QTabWidget();
  tabs->setTabPosition(QTabWidget::North);

  // General drawing settings
  {
    // Drawing source
    ValueCombo<DrawingSource>* comboDrawSource = new ValueCombo<DrawingSource>(m_settings.m_source, this);
    for (int i = DrawingSource_Begin; i < DrawingSource_End; i ++)
      comboDrawSource->add(DrawingSourceStr[i], (DrawingSource)i);

    // Drawing mode
    ValueCombo<DrawingMode>* comboDrawMode = new ValueCombo<DrawingMode>(m_settings.m_mode, this);
    for (int i = DrawingMode_Begin; i < DrawingMode_End; i ++)
      comboDrawMode->add(DrawingModeStr[i], (DrawingMode)i);

    // Histogram color
    ColorButton* buttonHistoOutlineColor = new ColorButton(m_settings.m_histoOutlineColor, this);
    ColorButton* buttonHistoBinColor = new ColorButton(m_settings.m_histoBinColor, this);
    FontColorButton* buttonHistoTextFont = new FontColorButton(m_settings.m_histoTextFont, m_settings.m_histoTextColor, this);

    // Ground truth points
    ValueCheck* checkGTBbxPoints = new ValueCheck(m_settings.m_gt_bbx, "GT bounding box");
    ColorButton* buttonGTBbxColor = new ColorButton(m_settings.m_gtBbxColor, this);
    ValueCheck* checkGTKeypoints = new ValueCheck(m_settings.m_gt_keypoints, "GT keypoints");
    FontColorButton* buttonGTKeypointsFont = new FontColorButton(m_settings.m_gtKeypointsFont, m_settings.m_gtKeypointsFontColor, this);
    ColorButton* buttonGTKeypointsColor = new ColorButton(m_settings.m_gtKeypointsColor, this);
    ValueCheck* checkGTKeyLabel = new ValueCheck(m_settings.m_gt_keyLabel, "GT keypoints label");
    ValueCheck* checkGTLabel = new ValueCheck(m_settings.m_gt_label, "GT label");
    FontColorButton* buttonGTLabelFont = new FontColorButton(m_settings.m_gtLabelFont, m_settings.m_gtLabelFontColor, this);
    ColorButton* buttonGTLabelColor = new ColorButton(m_settings.m_gtLabelColor, this);

    // Multi-block cell size
    ValueCombo<std::size_t>* comboMBCx = new ValueCombo<std::size_t>(m_settings.m_cx, this);
    ValueCombo<std::size_t>* comboMBCy = new ValueCombo<std::size_t>(m_settings.m_cy, this);

    for (std::size_t i = ItemSettings::min_cell_size(); i <= ItemSettings::max_cell_size(); i ++)
    {
      comboMBCx->add(QObject::tr("cx: %1").arg(i), i);
      comboMBCy->add(QObject::tr("cy: %1").arg(i), i);
    }

    // Build the tab
    QGroupBox* group = new QGroupBox("Drawing settings");
    QGridLayout* vLayout = new QGridLayout();
    int row = 0;
    vLayout->addWidget(new QLabel("Source"), row, 0, 1, 2);
    vLayout->addWidget(comboDrawSource, row, 2, 1, 4);
    row ++;
    vLayout->addWidget(new QLabel("Source"), row, 0, 1, 2);
    vLayout->addWidget(comboDrawMode, row, 2, 1, 4);
    row ++;
    vLayout->addWidget(hSeparator(), row, 0, 1, 6);
    row ++;
    vLayout->addWidget(checkGTBbxPoints, row, 0, 1, 4);
    vLayout->addWidget(buttonGTBbxColor, row, 2, 1, 4);
    row ++;
    vLayout->addWidget(checkGTLabel, row, 0, 1, 2);
    vLayout->addWidget(buttonGTLabelFont, row, 2, 1, 2);
    vLayout->addWidget(buttonGTLabelColor, row, 4, 1, 2);
    row ++;
    vLayout->addWidget(checkGTKeypoints, row, 0, 1, 2);
    vLayout->addWidget(buttonGTKeypointsColor, row, 2, 1, 4);		
    row ++;
    vLayout->addWidget(checkGTKeyLabel, row, 0, 1, 2);
    vLayout->addWidget(buttonGTKeypointsFont, row, 2, 1, 4);
    row ++;
    vLayout->addWidget(hSeparator(), row, 0, 1, 6);
    row ++;
    vLayout->addWidget(new QLabel("Histo (bins & outline):"), row, 0, 1, 2);
    vLayout->addWidget(buttonHistoBinColor, row, 2, 1, 2);
    vLayout->addWidget(buttonHistoOutlineColor, row, 4, 1, 2);
    row ++;
    vLayout->addWidget(new QLabel("Histo (text):"), row, 0, 1, 2);
    vLayout->addWidget(buttonHistoTextFont, row, 2, 1, 4);
    row ++;
    vLayout->addWidget(hSeparator(), row, 0, 1, 6);
    row ++;
    vLayout->addWidget(new QLabel("Multi-block cell size:"), row, 0, 1, 2);
    vLayout->addWidget(comboMBCx, row, 2, 1, 2);
    vLayout->addWidget(comboMBCy, row, 4, 1, 2);
    row ++;
    group->setLayout(vLayout);

    // Add the tab
    tabs->addTab(impl::buildTab(group), "Drawing");
  }

  // Text information settings
  QWidget* tiFrameWidget = impl::buildLayout(m_settings.m_textInfo, "Text information frame");
  {
    // Build the tab
    tabs->addTab(impl::buildTab(
          impl::buildGroupBox(m_settings.m_textInfo, "Text information")),
        "Text info");
  }

  // Text information frame settings
  {
    tabs->addTab(impl::buildTab(
          tiFrameWidget),
        "Text info frame");
  }

  // Build the main layout
  QVBoxLayout* mainLayout = new QVBoxLayout();
  mainLayout->addWidget(tabs);
  mainLayout->addLayout(buildOKCancelLayout(this));

  setLayout(mainLayout);
  setWindowTitle("Item Drawing Settings");
  //setFixedSize(QSize(480, 400));
  resize(QSize(480, 400));
  setModal(true);
}
