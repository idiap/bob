/**
 * @file visioner/cxx/cv_draw.cc
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

#include <QPainter>

#include "bob/visioner/cv/cv_draw.h"

namespace bob { namespace visioner {

  QImage draw_gt(const ipscale_t& ipscale)
  {
    QImage qimage = visioner::convert(ipscale.m_image);

    QPainter painter(&qimage);
    painter.setPen(QPen(QBrush(qRgb(0, 255, 0)), 2.0, Qt::SolidLine));

    // Draw each object
    for (std::vector<Object>::const_iterator ito = ipscale.m_objects.begin();
        ito != ipscale.m_objects.end(); ++ ito)
    {
      // Bounding box
      const Object& object = *ito;
      painter.drawRect(object.bbx());

      // Keypoints
      for (std::vector<bob::visioner::Keypoint>::const_iterator itk = object.keypoints().begin();
          itk != object.keypoints().end(); ++ itk)
      {
        const visioner::Keypoint& keypoint = *itk;
        const QPointF& point = keypoint.m_point;

        painter.drawLine(point.x() - 4, point.y(), point.x() + 4, point.y());
        painter.drawLine(point.x(), point.y() - 4, point.x(), point.y() + 4);
      }
    }

    return qimage;
  }

  void draw_detection(QImage& qimage, const detection_t& detection, const param_t& param, bool label)
  {
    QPainter painter(&qimage);

    QFont font = painter.font();
    font.setBold(true);
    painter.setFont(font);
    static const QPen pen_true(QBrush(qRgb(0, 0, 255)), 2.0, Qt::SolidLine);
    static const QPen pen_false(QBrush(qRgb(255, 0, 0)), 2.0, Qt::SolidLine);

    const double& score = detection.first;
    const QRectF& bbx = detection.second.first;                
    const int output = detection.second.second;

    // Detection: bounding box
    painter.setPen(label == true ? pen_true : pen_false);
    painter.drawRect(bbx);

    // Detection: score (3D)
    const QString text_score = QObject::tr("%1").arg(score, 0, 'f', 2);

    painter.setPen(qRgb(155, 155, 255));
    painter.drawText(bbx.left() + 1,
        bbx.top() + 1 + painter.fontMetrics().height(),
        text_score);

    painter.setPen(qRgb(105, 105, 255));
    painter.drawText(bbx.left() + 2,
        bbx.top() + 2 + painter.fontMetrics().height(),
        text_score);

    painter.setPen(qRgb(55, 55, 255));
    painter.drawText(bbx.left() + 3,
        bbx.top() + 3 + painter.fontMetrics().height(),
        text_score);

    // Detection: label
    painter.setPen(label == true ? pen_true : pen_false);
    const QString text_label = QObject::tr("%1").arg(param.m_labels[output].c_str());
    painter.drawText(bbx.right() - painter.fontMetrics().width(text_label) - 2, 
        bbx.bottom() - 2,
        text_label);                
  }

  void draw_detections(QImage& qimage, const std::vector<detection_t>& detections, const param_t& param, const std::vector<int>& labels)
  {
    if (detections.size() == labels.size())
    {
      for (uint64_t i = 0; i < detections.size(); i ++)
      {
        draw_detection(qimage, detections[i], param, labels[i]);
      }
    }
  }

  void draw_points(QImage& qimage, const std::vector<QPointF>& points)
  {
    QPainter painter(&qimage);

    painter.setPen(QPen(QBrush(qRgb(255, 0, 0)), 2.0, Qt::SolidLine));
    for (std::vector<QPointF>::const_iterator it = points.begin(); it != points.end(); ++ it)
    {
      const QPointF& point = *it;

      painter.drawLine(point.x() - 4, point.y(), point.x() + 4, point.y());
      painter.drawLine(point.x(), point.y() - 4, point.x(), point.y() + 4);
    }
  }

  void draw_label(QImage& qimage, const detection_t& detection, const param_t& param, 
      uint64_t gt_label, uint64_t dt_label)
  {
    QPainter painter(&qimage);

    QFont font = painter.font();
    font.setBold(true);
    painter.setFont(font);

    static const QPen pen_true(QBrush(qRgb(0, 0, 255)), 2.0, Qt::SolidLine);
    static const QPen pen_false(QBrush(qRgb(255, 0, 0)), 2.0, Qt::SolidLine);

    const QRectF& bbx = detection.second.first;                

    painter.setPen(gt_label == dt_label ? pen_true : pen_false);
    const QString text_label = QObject::tr("<%1> / <%2>")
      .arg(param.m_labels[dt_label].c_str())
      .arg(param.m_labels[gt_label].c_str());
    painter.drawText(bbx.left() + 2, 
        bbx.top() - 4 * painter.fontMetrics().height() / 5,
        text_label);    
  }

}}
