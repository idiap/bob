/**
 * @file visioner/programs/vgui/fmap_item.cc
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

#include <QTime>
#include <QGraphicsSceneMouseEvent>
#include <QCursor>
#include <functional>

#include "visioner/vision/mb_xlbp.h"
#include "visioner/vision/integral.h"

#include "fmap_item.h"
#include "image_collection.h"

namespace impl {
	// Feature codes mapping to RGB colors
	struct ToRgbs : public boost::serialization::singleton<ToRgbs>
	{
                ToRgbs() :	m_8bits2rgbs(256),
				m_9bits2rgbs(512)
		{              
                        // 8-bit features mapping to RGB
			for (int i = 0; i < 256; i ++)
			{
				const int grey = i;
				m_8bits2rgbs[i] = qRgb(grey, grey, grey);
			}
			
			// 9-bit features mapping to RGB
			for (int i = 0; i < 512; i ++)
			{
				const int grey = i * 256 / 512;
				m_9bits2rgbs[i] = qRgb(grey, grey, grey);
			}
		}
		
		// Attributes
                std::vector<QRgb>	m_8bits2rgbs;		// 8-bit features mapping to RGB
		std::vector<QRgb>	m_9bits2rgbs;		// 9-bit features mapping to RGB
	};
}

FeatureMapItem::FeatureMapItem(const SceneSettings& global, const ItemSettings& settings)
	:	m_global(global), m_settings(settings),
		m_src_colors(&impl::ToRgbs::get_const_instance().m_8bits2rgbs),
          	m_mouse(Ignore),
		m_pane_x(0.5), m_pane_y(0.5)
{
	setFlag(QGraphicsItem::ItemIsMovable);
	setFlag(QGraphicsItem::ItemIsSelectable);
	setFlag(QGraphicsItem::ItemIsFocusable);
	setCacheMode(QGraphicsItem::NoCache);
	setPos(settings.m_scene_pos);
}

void FeatureMapItem::setSettings(const ItemSettings& settings)
{
	update_fmap(settings);
	update();
}

bool FeatureMapItem::setSource(DrawingSource source)
{
	ItemSettings settings = m_settings;
	settings.m_source = source;
	return update_fmap(settings);
}

bool FeatureMapItem::setMode(DrawingMode mode)
{
	ItemSettings settings = m_settings;
	settings.m_mode = mode;
	return update_fmap(settings);
}
	
bool FeatureMapItem::update_fmap(const ItemSettings& settings)
{
	const bool settings_changed = m_settings != settings;
	m_settings = settings;
	
	const bob::visioner::ipscale_t& ipscale = ImageCollection::get_const_instance().ipscale();
	if (settings_changed == true || ipscale.m_image != m_src_image)
	{
		update_fmap();
		return true;
	}

	return false;
}

void FeatureMapItem::update_fmap()
{
	const bob::visioner::ipscale_t& ipscale = ImageCollection::get_const_instance().ipscale();
	
	m_src_image = ipscale.m_image;
	bob::visioner::integral(m_src_image, m_src_iimage);

	// Decode the drawing source
	const impl::ToRgbs& theFMap2Rgbs = impl::ToRgbs::get_const_instance();
	m_src_colors = &theFMap2Rgbs.m_8bits2rgbs;
	switch (m_settings.m_source)
	{
                // MB-LBP code map
        case DrawingSource_LBP:
                bob::visioner::mb_dense<
                        uint32_t, uint16_t, 3, 3, 
                        bob::visioner::mb_lbp<uint32_t, uint16_t> >(
                        m_src_iimage, m_settings.m_cx, m_settings.m_cy, m_src_fmap);
                m_src_colors = &theFMap2Rgbs.m_8bits2rgbs;
		break;
                
                // MB-mLBP code map
        case DrawingSource_mLBP:
                bob::visioner::mb_dense<
                        uint32_t, uint16_t, 3, 3, 
                        bob::visioner::mb_mlbp<uint32_t, uint16_t> >(
                        m_src_iimage, m_settings.m_cx, m_settings.m_cy, m_src_fmap);
                m_src_colors = &theFMap2Rgbs.m_8bits2rgbs;
		break;                
                
                // MB-tLBP code map
        case DrawingSource_tLBP:
                bob::visioner::mb_dense<
                        uint32_t, uint16_t, 3, 3, 
                        bob::visioner::mb_tlbp<uint32_t, uint16_t> >(
                        m_src_iimage, m_settings.m_cx, m_settings.m_cy, m_src_fmap);
                m_src_colors = &theFMap2Rgbs.m_8bits2rgbs;
		break;
                
                // MB-dLBP code map
        case DrawingSource_dLBP:
                bob::visioner::mb_dense<
                        uint32_t, uint16_t, 3, 3,
                        bob::visioner::mb_dlbp<uint32_t, uint16_t> >(
                        m_src_iimage, m_settings.m_cx, m_settings.m_cy, m_src_fmap);
                m_src_colors = &theFMap2Rgbs.m_8bits2rgbs;
		break;
                
		// Greyscale values
	case DrawingSource_Greys:
	default:
		m_src_fmap = m_src_image;
		m_src_colors = &theFMap2Rgbs.m_8bits2rgbs;
		break;
	}	
	
	// Build the Qt feature map
	const std::vector<QRgb>& colors = *m_src_colors;
	m_src_qfmap = QImage(m_src_fmap.cols(), m_src_fmap.rows(), QImage::Format_RGB32);
	for (int j = 0, y = 0; j < m_src_qfmap.height(); j ++, y ++)
	{
		const uint16_t* p_src = &m_src_fmap[j][0];
		for (int i = 0, x = 0; i < m_src_qfmap.width(); i ++, x ++)
		{
			const QRgb color = colors[*(p_src ++)];
			m_src_qfmap.setPixel(x, y, color);
		}
	}

	// Build the feature map histogram
	const int src_max = m_src_colors->size();
	m_src_histo.reset(src_max, 0, src_max - 1);
	m_src_histo.add(m_src_fmap.begin(), m_src_fmap.end());
	if (m_settings.m_mode == DrawingMode_CumHistogram)
	{
		m_src_histo.cumulate();
	}
	
//	// DEBUG: 	
//	{
//		int cnt = 0;
//		for (bob::visioner::Matrix<uint16_t>::iterator_t it = m_src_fmap.begin(); it != m_src_fmap.end(); ++ it)
//		{
//			if ((*it) < 0 || (*it) > src_max)
//			{
//				//std::cout << "*it = " << (*it) << "\n";
//				cnt ++;
//				//exit(EXIT_FAILURE);
//			}
//		}
//		std::cout << "out of range: cnt = " << cnt << "/" << m_src_fmap.size() << "\n";
//	}
}

QRectF FeatureMapItem::boundingRect() const
{
	return QRectF(-m_settings.m_scene_size.width() / 2, 
	              -m_settings.m_scene_size.height() / 2, 
	              m_settings.m_scene_size.width(), m_settings.m_scene_size.height());
}

QRectF FeatureMapItem::imageRect() const
{
	return QRectF(- m_pane_x * m_src_fmap.cols(), - m_pane_y * m_src_fmap.rows(),
		      m_src_fmap.cols(), m_src_fmap.rows());
}

QRectF FeatureMapItem::drawingRect() const
{
	return boundingRect().intersected(imageRect());
}

void FeatureMapItem::setSize(qreal width, qreal height)
{
	m_settings.m_scene_size = QSizeF(width, height);
	prepareGeometryChange();
}

void FeatureMapItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
	static QTime drawingTime;
	drawingTime.restart();
	
	update_fmap(m_settings);
	
	// Draw the background
	const FrameSettings& frame = isSelected() ? m_global.m_selFrame : m_global.m_notSelFrame;
	frame.drawBackground(*painter, boundingRect());

	// Decode the drawing mode
	switch (m_settings.m_mode)
	{
		// Draw the feature map histogram
	case DrawingMode_Histogram:
	case DrawingMode_CumHistogram:
		drawHistogram(*painter);
		break;

		// Draw the feature map and the ground truth points
	default:
		drawMap(*painter);
		drawGTruth(*painter);
		break;
	}
	
        // Draw border
        frame.drawBorder(*painter, boundingRect());

	// Update and draw the text information if required
	if (	m_settings.m_textInfo.m_enable == true &&
		m_settings.m_textInfo.anyLineEnabled() == true)
	{
		// Drawing source
		m_settings.m_textInfo.update(0,
			QObject::tr(":: %1 [%2x%3]")
                                .arg(DrawingSourceStr[m_settings.m_source])
                                .arg(m_settings.m_cx).arg(m_settings.m_cy));
	
		// Source image name
		m_settings.m_textInfo.update(1,
			QObject::tr(":: %1").arg(ImageCollection::get_const_instance().name().c_str()));
	
		// Drawing time
		m_settings.m_textInfo.update(2,
			QObject::tr("[%1 ms]").arg(drawingTime.elapsed()));

		drawTextInfos(*painter);
	}
}

void FeatureMapItem::mousePressEvent(QGraphicsSceneMouseEvent* event)
{
        QGraphicsItem::CacheMode old_cachemode = cacheMode();
	setCacheMode(QGraphicsItem::DeviceCoordinateCache);
	
	m_mouse = Ignore;
	
	// Resizing & paning
	if (	(event->buttons() & Qt::LeftButton) == Qt::LeftButton &&
		(event->modifiers() & Qt::ControlModifier) == Qt::ControlModifier)
	{
		if (FrameSettings::bottomLeftResizable(boundingRect(), event->pos()))
		{
			m_mouse = BottomLeftResize;
			setCursor(Qt::SizeBDiagCursor);
		}
		else if (FrameSettings::bottomRightResizable(boundingRect(), event->pos()))
		{
			m_mouse = BottomRightResize;
			setCursor(Qt::SizeFDiagCursor);
		}
		else if (FrameSettings::topLeftResizable(boundingRect(), event->pos()))
		{
			m_mouse = TopLeftResize;
			setCursor(Qt::SizeFDiagCursor);
		}
		else if (FrameSettings::topRightResizable(boundingRect(), event->pos()))
		{
			m_mouse = TopRightResize;
			setCursor(Qt::SizeBDiagCursor);
		}
		else if (m_settings.m_mode == DrawingMode_Original)
		{
			m_mouse = Pane;
			setCursor(Qt::ClosedHandCursor);
		}
	}

	// Zooming
	else if ((event->buttons() & Qt::LeftButton) == Qt::LeftButton &&
		(event->modifiers() & Qt::ShiftModifier) == Qt::ShiftModifier)
	{
		m_mouse = Zoom;
		setCursor(Qt::ArrowCursor);
	}
	
        setCacheMode(old_cachemode);
	QGraphicsItem::mousePressEvent(event);
}

void FeatureMapItem::mouseReleaseEvent(QGraphicsSceneMouseEvent* event)
{
	m_mouse = Ignore;
	setCursor(Qt::ArrowCursor);
	setCacheMode(QGraphicsItem::NoCache);
	
	QGraphicsItem::mouseReleaseEvent(event);
}

void FeatureMapItem::mouseMoveEvent(QGraphicsSceneMouseEvent* event)
{
	m_settings.m_scene_pos = scenePos();
	
	// Resizing & paning
	if (	(event->buttons() & Qt::LeftButton) == Qt::LeftButton &&
		(event->modifiers() & Qt::ControlModifier) == Qt::ControlModifier &&
		m_mouse != Ignore)
	{
		// Paning
		if (m_mouse == Pane)
		{
			event->accept();
						
			const qreal dx = event->scenePos().x() - event->lastScenePos().x();
			const qreal dy = event->scenePos().y() - event->lastScenePos().y();			
			
			m_pane_x -= dx * bob::visioner::inverse(m_src_fmap.cols());
			m_pane_y -= dy * bob::visioner::inverse(m_src_fmap.rows());
			
			update();
		}
		
		// Resizing
		else
		{
			event->accept();
			
			const qreal dx = event->scenePos().x() - event->lastScenePos().x();
			const qreal dy = event->scenePos().y() - event->lastScenePos().y();			
			
			QRectF brect = mapToScene(boundingRect()).boundingRect();
			qreal top = brect.top(), bottom = brect.bottom();
			qreal left = brect.left(), right = brect.right();
			
			switch (m_mouse)
			{
			case BottomLeftResize:
				bottom += dy, left += dx;
				break;
				
			case BottomRightResize:
				bottom += dy, right += dx;
				break;
				
			case TopLeftResize:
				top += dy, left += dx;
				break;
				
			case TopRightResize:
				top += dy, right += dx;
				break;
	
			default:
				break;
			}
			
			brect = QRectF(std::min(left, right), std::min(top, bottom),
			               std::max(left, right) - std::min(left, right),
			               std::max(bottom, top) - std::min(bottom, top));
			if (brect.width() >= 64 && brect.height() >= 48)
			{
				setPos(brect.center());
			
				brect = mapFromScene(brect).boundingRect();			
				setSize(brect.width(), brect.height());
			}
		}		
		return;
	}

	// Zooming
	else if (	(event->buttons() & Qt::LeftButton) == Qt::LeftButton &&
			(event->modifiers() & Qt::ShiftModifier) == Qt::ShiftModifier &&
			m_mouse == Zoom)
	{
		const qreal dy = event->scenePos().y() - event->lastScenePos().y();
		setScale(scale() * (dy > 0.0 ? 1.1 : 0.9));
	}
	
	QGraphicsItem::mouseMoveEvent(event);
}

void FeatureMapItem::drawMap(QPainter& painter)
{
	if (m_src_fmap.empty() == true)
	{
		return;
	}

	painter.drawImage(drawingRect().topLeft(),
			  m_src_qfmap,
			  drawingRect().translated(m_pane_x * m_src_fmap.cols(), m_pane_y * m_src_fmap.rows()));
}

void FeatureMapItem::drawGTruth(QPainter& painter)
{
	const bob::visioner::ipscale_t& ipscale = ImageCollection::get_const_instance().ipscale();
	
	// Prepare displacement factors
	const qreal dx = imageRect().left();
	const qreal dy = imageRect().top();
	
	// Draw each object: bounding box + keypoints + label
	for (std::vector<bob::visioner::Object>::const_iterator it = ipscale.m_objects.begin();
		it != ipscale.m_objects.end(); ++ it)
	{
		const bob::visioner::Object& object = *it;
		const QRectF& bbx = object.bbx();

		QRectF draw_bbx = bbx.translated(dx, dy);

		// Bounding box
		if (	m_settings.m_gt_bbx == true &&
			boundingRect().contains(draw_bbx) == true)
		{
			painter.setPen(QPen(m_settings.m_gtBbxColor, 2, Qt::SolidLine));
			painter.drawRect(draw_bbx);
		}

		// Label
		if (m_settings.m_gt_label == true)
		{
			painter.setFont(m_settings.m_gtLabelFont);

			const QFontMetrics fm = painter.fontMetrics();
			const int text_h = fm.height() * 2 / 3;
			const int tx = 2, ty = 2;

			QRect rect(draw_bbx.left(), draw_bbx.top() - text_h - 3 * ty,
				   draw_bbx.width(), text_h + 2 * ty);

			if (boundingRect().contains(rect) == true)
			{
				painter.fillRect(rect, m_settings.m_gtLabelColor);

				painter.setPen(m_settings.m_gtLabelFontColor);
				painter.drawText(draw_bbx.left() + tx, draw_bbx.top() - ty,
                                        QObject::tr("%1/%2/%3")
                                        .arg(object.type().c_str())
                                        .arg(object.pose().c_str())
                                        .arg(object.id().c_str()));
			}
		}
		
		// Keypoints
		for (std::vector<bob::visioner::Keypoint>::const_iterator itf = object.keypoints().begin();
			itf != object.keypoints().end(); ++ itf)
		{
			static const int delta = 6;
			
			painter.setPen(QPen(m_settings.m_gtKeypointsColor, 3, Qt::SolidLine));
			const qreal x = dx + itf->m_point.x();
			const qreal y = dy + itf->m_point.y();
			
			if (boundingRect().contains(QRectF(x - delta, y - delta, 2 * delta, 2 * delta)) == true)
			{
                                if (m_settings.m_gt_keypoints == true)
                                {
                                        painter.drawLine(x - delta, y, x + delta, y);
                                        painter.drawLine(x, y - delta, x, y + delta);	
                                }
                                
                                if (m_settings.m_gt_keyLabel == true)
                                {	
                                        painter.setFont(m_settings.m_gtKeypointsFont);
                                        painter.setPen(m_settings.m_gtKeypointsFontColor);
                                        painter.drawText(x, y, QObject::tr("%1").arg(itf->m_id.c_str()));
                                }
			}
		}
	}
}

void FeatureMapItem::drawHistogram(QPainter& painter)
{
	const std::vector<double>& bins = m_src_histo.bins();
	const float max_bin = bins.empty() ? 1.0 : *std::max_element(bins.begin(), bins.end());
	const int n_bins = bins.size();
	
	// Set the font for the maximum value
	painter.setFont(m_settings.m_histoTextFont);
	QFontMetrics fontMetrics = painter.fontMetrics();

	const int bx = 6, by = 6;				// Borders
	const int sy = 12;					// Gradient height

	//	... histogram area
	const QRect histo_area(	boundingRect().left() + bx,
				boundingRect().top() + 2 * by + fontMetrics.height(),
				boundingRect().width() - 2 * bx,
				boundingRect().height() - 4 * by - sy - fontMetrics.height());
	const float dh = (float)histo_area.height() / max_bin;
	const float dw = (float)histo_area.width() * bob::visioner::inverse(n_bins);

	//	... text point
	const QPoint text_point(histo_area.left(), histo_area.top() - by);
	
	//	... gradient area
	const QRect grad_area(histo_area.left(), histo_area.top() + histo_area.height() + by,
			      histo_area.width(), sy);

	// Draw every bin
	QBrush binBrush(m_settings.m_histoBinColor, Qt::SolidPattern);
	for (int px = 0; px < n_bins; px ++)
	{
		const float x = px * dw + histo_area.left(), dx = dw;
		const float dy = dh * bins[px], y = histo_area.bottom() - dy;
		painter.fillRect(floor(x), floor(y), ceil(dx), ceil(dy), binBrush);
	}

	// Draw the gradient for the specified color channel
	const std::vector<QRgb>& colors = *m_src_colors;
	QLinearGradient gradient(0, 0, 1, 0);
	gradient.setCoordinateMode(QGradient::ObjectBoundingMode);
	gradient.setColorAt(0.0, colors[0]);
	gradient.setColorAt(1.0, colors[n_bins - 1]);
	painter.fillRect(grad_area, gradient);

        // Draw the gradient border
        painter.setPen(QPen(QBrush(m_settings.m_histoOutlineColor), 1, Qt::SolidLine, Qt::SquareCap));
        painter.setBrush(Qt::NoBrush);
        painter.drawRect(grad_area);

        // Draw the number of bins
        QString str = QObject::tr("[%1 bins]").arg(n_bins);
        painter.setPen(QPen(m_settings.m_histoTextColor, 1, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
        painter.drawText(text_point, str);
}

void FeatureMapItem::drawTextInfos(QPainter& painter)
{
        // For each text info, compute its bounding box using its font
        int text_width = 0, text_height = 0;
        for (TextInfoSettings::ConstItLine it = m_settings.m_textInfo.m_lines.begin();
                it != m_settings.m_textInfo.m_lines.end(); ++ it)
                        if (true == it->m_enable)
        {
                QFontMetrics fontMetrics(it->m_font);
                text_height += fontMetrics.height();
                text_width = std::max(text_width, fontMetrics.width(it->m_text));
        }

        const int out_border = 4, in_border = 4;
        if (	text_width + 4 * out_border >= boundingRect().width() ||
                text_height + 4 * in_border >= boundingRect().height())
                return;

        const int frame_width = text_width + 2 * in_border;
        const int frame_height = text_height + 2 * in_border;

        // Compute the bounding rect, based on text size and the selected position
        QRect frameRect;
        switch (m_settings.m_textInfo.m_position)
        {
        case TextInfoPos_TopLeft:
                frameRect = QRect(  	boundingRect().left() + out_border,
                                        boundingRect().top() + out_border,
                                        frame_width, frame_height);
                break;

        case TextInfoPos_TopRight:
                frameRect = QRect(	boundingRect().right() - out_border - frame_width,
                                        boundingRect().top() + out_border,
                                        frame_width, frame_height);
                break;

        case TextInfoPos_BottomLeft:
                frameRect = QRect(  	boundingRect().left() + out_border,
                                        boundingRect().bottom() - out_border - frame_height,
                                        frame_width, frame_height);
                break;

        case TextInfoPos_BottomRight:
                frameRect = QRect(  	boundingRect().right() - out_border - frame_width,
                                        boundingRect().bottom() - out_border - frame_height,
                                        frame_width, frame_height);
                break;

        default:
                return;
        }

	// Draw the frame
	m_settings.m_textInfo.m_frame.drawBackground(painter, frameRect);
	m_settings.m_textInfo.m_frame.drawBorder(painter, frameRect);

        // Draw the text lines in the frame
        const int x = frameRect.left() + in_border;
        int y = frameRect.top();
        for (TextInfoSettings::ConstItLine it = m_settings.m_textInfo.m_lines.begin();
                it != m_settings.m_textInfo.m_lines.end(); ++ it)
                        if (true == it->m_enable)
        {
                QFontMetrics fontMetrics(it->m_font);
                y += fontMetrics.height();

		painter.setFont(it->m_font);
		painter.setPen(it->m_color);
		painter.drawText(x, y, it->m_text);
	}
}

void FeatureMapItem::draw2DPlot(QPainter& painter, double value_min, double value_max,
                                const std::vector<double>& values)
{
//	if (values.size() < 2)
//	{
//		return;
//	}

//	// Set the font for the minimum and maximum stored values
//	painter.setFont(m_settings.m_histoTextFont);
//	QFontMetrics fontMetrics = painter.fontMetrics();

//	// Get the drawing dimensions
//	const QRect& drawRect = m_display.canvasArea();
//	const int bx = 6;									// Borders
//	const int by = 6;
//	const int max_liney = (drawRect.height() - 2 * by - 2 * fontMetrics.height());		// Maximum variation height
//	const int end_liney =  drawRect.bottom() - by - fontMetrics.height();			// Lowest Oy used for graphics
//	const int beginx = drawRect.left() + bx;
//	const int no_points = std::min((int)values.size(), drawRect.width() - 2 * bx);		// Number of points to display
//	const double linex = (drawRect.width() - 2 * bx + 0.0) / (no_points - 1.0);
//	const double inv_value = value_max == value_min ? 1.0 : 1.0 / (value_max - value_min);

//	// Draw the Ox axes
//	painter.drawLine(beginx, end_liney, beginx + (int)((no_points - 1) * linex), end_liney);

//	// Draw the dashing line from the graphic points to the Ox axes
//	painter.setPen(QPen(m_settings.m_histoOutlineColor, 1, Qt::DashLine, Qt::RoundCap, Qt::RoundJoin));
//	for (int i = 0; i < no_points; i ++)
//	{
//		const int x = beginx + (int)(i * linex);
//		const int y = end_liney - (int)((values[i] - value_min) * inv_value * max_liney);
//		painter.drawLine(x, y, x, end_liney);
//	}

//	// Draw the connecting lines - main graphic
//	painter.setPen(QPen(m_settings.m_histoOutlineColor, 1, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
//	painter.setBrush(m_settings.m_histoBinColor);
//	int last_y = by;
//	for (int i = 0; i < no_points; i ++)
//	{
//		const int y = end_liney - (int)((values[i] - value_min) * inv_value * max_liney);
//		if (i > 0)
//		{
//			painter.drawLine(beginx + (int)((i - 1) * linex), last_y, beginx + (int)(i * linex), y);
//		}
//		last_y = y;
//	}

//	// Draw the minimum and the maximum values
//	QString str = QObject::tr("#%1 values: [%1 - %2]")
//				.arg(values.size()).arg(value_min, 0, 'f', 2).arg(value_max, 0, 'f', 2);
//	painter.setPen(QPen(m_settings.m_histoTextColor, 1, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
//	painter.drawText(beginx, drawRect.top() + by + fontMetrics.height() / 2, str);

//	// Draw the first and the last value
//	for (int i = 0; i < no_points; i += no_points - 1)
//	{
//		const int x = beginx + (int)(i * linex);
//		const int y = end_liney;

//		str = QObject::tr("%1").arg(values[i], 0, 'f', 2);
//		painter.setPen(QPen(m_settings.m_histoTextColor, 1, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
//		painter.drawText(x - (i == 0 ? 0 : fontMetrics.width(str)), y + fontMetrics.height(), str);
//	}
}

void FeatureMapItem::drawArrow(QPainter& painter, int startx, int starty, int stopx, int stopy)
{
//	const double arrow_angle = M_PI / 12.0;
//	const double arrow_size = 8.0;

//	// Draw the connection line
//	painter.drawLine(startx, starty, stopx, stopy);

//	// Draw the arrow at the end of the connection line
//	//
//	// HowTo:	compute the arrow's points as the line was vertical and then multiply
//	//		with the rotation matrix of the line facing axes
//	//
//	const double angle = M_PI * 0.5 - std::atan2((double)(stopy - starty), (double)(stopx - startx));

//	QPoint arrow[3];
//	arrow[0].setX(stopx);
//	arrow[0].setY(stopy);

//	arrow[1].setX((int)(0.5 + stopx + arrow_size * sin(arrow_angle - angle)));
//	arrow[1].setY((int)(0.5 + stopy - arrow_size * cos(arrow_angle - angle)));

//	arrow[2].setX((int)(0.5 + stopx - arrow_size * sin(arrow_angle + angle)));
//	arrow[2].setY((int)(0.5 + stopy - arrow_size * cos(arrow_angle + angle)));

//	painter.drawPolygon(arrow, 3);
}
