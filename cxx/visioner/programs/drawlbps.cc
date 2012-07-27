/**
 * @file visioner/programs/drawlbps.cc
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

#include <QApplication>
#include <QPainter>
#include <QColor>
#include <boost/program_options.hpp>

#include "visioner/util/util.h"

////////////////////////////////////////////////////////////////////////////////////////
// Drawing functions for displaying the:
//      LBP, tLBP (+4x2,2x4), dLBP, mLBP (+4x2,2x4), MCT operators
////////////////////////////////////////////////////////////////////////////////////////

static int lbp3x3_indices[9] = { 0, 7, 6, 1, 8, 5, 2, 3, 4 };
static int lbp4x2_indices[8] = { 0, 7, 1, 6, 2, 5, 3, 4 };
static int lbp2x4_indices[8] = { 0, 7, 6, 5, 1, 2, 3, 4 };
static int lbp8x1_indices[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
static int lbp1x8_indices[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };

QImage draw_init(int bcx, int bcy, int ncx, int ncy) {
  return QImage(ncx * bcx, ncy * bcy, QImage::Format_RGB32);
}

void draw_base(QImage& image, int bcx, int bcy, int ncx, int ncy, bool darker, bool center) {

  // Draw the background
  QPainter painter(&image);
  painter.fillRect(painter.window(),
      darker == false ? QColor(255, 255, 255) : QColor(125, 125, 125));

  if (darker == true && center == false)
  {
    painter.fillRect(QRect(bcx, bcy, bcx, bcy), QColor(255, 255, 255));
  }

  const QPen pen_grid(QColor(175, 175, 175), std::max(2, bcx / 8));
  const QPen pen_text(QBrush(QColor(55, 55, 55)), std::max(2, bcx / 8));
  const QPen pen_cross(QBrush(QColor(255, 55, 55)), std::max(2, bcx / 16));

  QFont font = painter.font();
  font.setBold(true);
  font.setPixelSize(std::max(bcx, bcy) / 2);
  painter.setFont(font);

  // Draw the grid and the pixel indexes
  for (int x = 0, cnt = 0; x < ncx; x ++)
  {
    for (int y = 0; y < ncy; y ++, cnt ++)
    {
      painter.setPen(pen_grid);
      painter.drawRect(x * bcx, y * bcy, bcx, bcy);

      painter.setPen(pen_text);
      painter.drawText(QRect(x * bcx, y * bcy, bcx, bcy), Qt::AlignCenter, 
          QObject::tr("%1").arg(
            (ncx == 3 && ncy == 3 ? lbp3x3_indices[cnt] :
             (ncx == 4 && ncy == 2 ? lbp4x2_indices[cnt] :
              (ncx == 2 && ncy == 4 ? lbp2x4_indices[cnt] :
               (ncx == 8 && ncy == 1 ? lbp8x1_indices[cnt] : lbp1x8_indices[cnt]))))));

      if (center == false && x == ncx / 2 && y == ncy / 2)
      {
        const int dx = bcx / 4, dy = bcy / 4;

        painter.setPen(pen_cross);
        painter.drawLine(x * bcx + dx, y * bcy + dy, x * bcx + bcx - dx, y * bcy + bcy - dy);
        painter.drawLine(x * bcx + bcx - dx, y * bcy + dy, x * bcx + dx, y * bcy + bcy - dy);
      }
    }
  }
}

enum Direction
{
  N, S, E, W,
  NE, NW, SE, SW
};

void draw_connect(QImage& image, int bcx, int bcy, int startx, int starty, int stopx, int stopy)
{
  QPainter painter(&image);

  const QPen pen_line(QColor(75, 75, 75), std::max(2, bcx / 16));
  painter.setPen(pen_line);

  const double arrow_angle = M_PI / 8.0;
  const double arrow_size = std::min(bcx, bcy) / 6;

  // Draw the connection line
  painter.drawLine(startx, starty, stopx, stopy);

  // Draw the arrow at the end of the connection line
  //
  // HowTo:	compute the arrow's points as the line was vertical and then multiply
  //		with the rotation matrix of the line facing axes
  //
  const double angle = M_PI * 0.5 - std::atan2((double)(stopy - starty), (double)(stopx - startx));

  QPoint arrow[3];
  arrow[0].setX(stopx);
  arrow[0].setY(stopy);

  arrow[1].setX((int)(0.5 + stopx + arrow_size * sin(arrow_angle - angle)));
  arrow[1].setY((int)(0.5 + stopy - arrow_size * cos(arrow_angle - angle)));

  arrow[2].setX((int)(0.5 + stopx - arrow_size * sin(arrow_angle + angle)));
  arrow[2].setY((int)(0.5 + stopy - arrow_size * cos(arrow_angle + angle)));

  painter.drawPolygon(arrow, 3);
}

void draw_connect(QImage& image, int bcx, int bcy, int x, int y, Direction dir)
{
  const int rx = bcx / 6, ry = bcy / 6;
  const int dx = x * bcx, dy = y * bcy;

  switch (dir)
  {
    case N:         
      draw_connect(image, bcx, bcy, dx + bcx / 2, dy + ry, dx + bcx / 2, dy - ry);
      break;

    case S:         
      draw_connect(image, bcx, bcy, dx + bcx / 2, dy + bcy - ry, dx + bcx / 2, dy + bcy + ry);
      break;

    case E:         
      draw_connect(image, bcx, bcy, dx + bcx - rx, dy + bcy / 2, dx + bcx + rx, dy + bcy / 2);
      break;

    case W:         
      draw_connect(image, bcx, bcy, dx + rx, dy + bcy / 2, dx - rx, dy + bcy / 2);
      break;

    case NE:         
      draw_connect(image, bcx, bcy, dx + bcx - rx, dy + ry, dx + bcx + rx, dy - ry);
      break;

    case NW:         
      draw_connect(image, bcx, bcy, dx + rx, dy + ry, dx - rx, dy - ry);
      break;

    case SE:         
      draw_connect(image, bcx, bcy, dx + bcx - rx, dy + bcy - ry, dx + bcx + rx, dy + bcy + ry);
      break;

    case SW:         
      draw_connect(image, bcx, bcy, dx + rx, dy + bcy - ry, dx - rx, dy + bcy + ry);
      break;

    default:
      break;
  }
}

void draw_lbp(QImage& image, int bcx, int bcy)
{
  draw_base(image, bcx, bcy, 3, 3, false, false);

  // Draw the binary associations
  draw_connect(image, bcx, bcy, 0, 0, SE);
  draw_connect(image, bcx, bcy, 1, 0, S);
  draw_connect(image, bcx, bcy, 2, 0, SW);
  draw_connect(image, bcx, bcy, 2, 1, W);
  draw_connect(image, bcx, bcy, 2, 2, NW);
  draw_connect(image, bcx, bcy, 1, 2, N);
  draw_connect(image, bcx, bcy, 0, 2, NE);
  draw_connect(image, bcx, bcy, 0, 1, E);
}

void draw_dlbp(QImage& image, int bcx, int bcy)
{
  draw_base(image, bcx, bcy, 3, 3, false, false);

  // Draw the binary associations
  draw_connect(image, bcx, bcy, 0, 0, SE);
  draw_connect(image, bcx, bcy, 1, 1, SE);
  draw_connect(image, bcx, bcy, 1, 0, S);
  draw_connect(image, bcx, bcy, 1, 1, S);
  draw_connect(image, bcx, bcy, 2, 0, SW);
  draw_connect(image, bcx, bcy, 1, 1, SW);
  draw_connect(image, bcx, bcy, 2, 1, W);
  draw_connect(image, bcx, bcy, 1, 1, W);
}

void draw_tlbp(QImage& image, int bcx, int bcy)
{
  draw_base(image, bcx, bcy, 3, 3, false, false);

  // Draw the binary associations
  draw_connect(image, bcx, bcy, 0, 0, E);
  draw_connect(image, bcx, bcy, 1, 0, E);
  draw_connect(image, bcx, bcy, 2, 0, S);
  draw_connect(image, bcx, bcy, 2, 1, S);
  draw_connect(image, bcx, bcy, 2, 2, W);
  draw_connect(image, bcx, bcy, 1, 2, W);
  draw_connect(image, bcx, bcy, 0, 2, N);
  draw_connect(image, bcx, bcy, 0, 1, N);
}

void draw_tlbp4x2(QImage& image, int bcx, int bcy)
{
  draw_base(image, bcx, bcy, 4, 2, false, true);

  // Draw the binary associations
  draw_connect(image, bcx, bcy, 0, 0, E);
  draw_connect(image, bcx, bcy, 1, 0, E);
  draw_connect(image, bcx, bcy, 2, 0, E);
  draw_connect(image, bcx, bcy, 3, 0, S);
  draw_connect(image, bcx, bcy, 3, 1, W);
  draw_connect(image, bcx, bcy, 2, 1, W);
  draw_connect(image, bcx, bcy, 1, 1, W);
  draw_connect(image, bcx, bcy, 0, 1, N);
}

void draw_tlbp8x1(QImage& image, int bcx, int bcy)
{
  draw_base(image, bcx, bcy, 8, 1, false, true);

  // Draw the binary associations
  draw_connect(image, bcx, bcy, -1, 0, E);
  draw_connect(image, bcx, bcy, 0, 0, E);
  draw_connect(image, bcx, bcy, 1, 0, E);
  draw_connect(image, bcx, bcy, 2, 0, E);
  draw_connect(image, bcx, bcy, 3, 0, E);
  draw_connect(image, bcx, bcy, 4, 0, E);
  draw_connect(image, bcx, bcy, 5, 0, E);
  draw_connect(image, bcx, bcy, 6, 0, E);
  draw_connect(image, bcx, bcy, 7, 0, E);
}

void draw_tlbp1x8(QImage& image, int bcx, int bcy)
{
  draw_base(image, bcx, bcy, 1, 8, false, true);

  // Draw the binary associations
  draw_connect(image, bcx, bcy, 0, -1, S);
  draw_connect(image, bcx, bcy, 0, 0, S);
  draw_connect(image, bcx, bcy, 0, 1, S);
  draw_connect(image, bcx, bcy, 0, 2, S);
  draw_connect(image, bcx, bcy, 0, 3, S);
  draw_connect(image, bcx, bcy, 0, 4, S);
  draw_connect(image, bcx, bcy, 0, 5, S);
  draw_connect(image, bcx, bcy, 0, 6, S);
  draw_connect(image, bcx, bcy, 0, 7, S);
}

void draw_tlbp2x4(QImage& image, int bcx, int bcy)
{
  draw_base(image, bcx, bcy, 2, 4, false, true);

  // Draw the binary associations
  draw_connect(image, bcx, bcy, 0, 0, E);
  draw_connect(image, bcx, bcy, 1, 0, S);
  draw_connect(image, bcx, bcy, 1, 1, S);
  draw_connect(image, bcx, bcy, 1, 2, S);
  draw_connect(image, bcx, bcy, 1, 3, W);
  draw_connect(image, bcx, bcy, 0, 3, N);
  draw_connect(image, bcx, bcy, 0, 2, N);
  draw_connect(image, bcx, bcy, 0, 1, N);
}

void draw_mct(QImage& image, int bcx, int bcy)
{
  draw_base(image, bcx, bcy, 3, 3, true, true);
}

void draw_mlbp(QImage& image, int bcx, int bcy)
{
  draw_base(image, bcx, bcy, 3, 3, true, false);
}

void draw_mlbp4x2(QImage& image, int bcx, int bcy)
{
  draw_base(image, bcx, bcy, 4, 2, true, true);
}

void draw_mlbp2x4(QImage& image, int bcx, int bcy)
{
  draw_base(image, bcx, bcy, 2, 4, true, true);
}

void draw_mlbp8x1(QImage& image, int bcx, int bcy)
{
  draw_base(image, bcx, bcy, 8, 1, true, true);
}

void draw_mlbp1x8(QImage& image, int bcx, int bcy)
{
  draw_base(image, bcx, bcy, 1, 8, true, true);
}

int main(int argc, char *argv[]) {	

  QApplication app(argc, argv);
  Q_UNUSED(app);

  // Parse the command line
  boost::program_options::options_description po_desc("", 160);
  po_desc.add_options()
    ("help,h", "help message");
  po_desc.add_options()
    ("bcx", boost::program_options::value<int>()->default_value(128),
     "board: cell width in pixels")
    ("bcy", boost::program_options::value<int>()->default_value(128),
     "board: cell height in pixels");

  boost::program_options::variables_map po_vm;
  boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
      .options(po_desc).run(),
      po_vm);
  boost::program_options::notify(po_vm);

  // Check arguments and options
  if (	po_vm.empty() || po_vm.count("help"))
  {
    bob::visioner::log_error("drawmbs") << po_desc << "\n";
    exit(EXIT_FAILURE);
  }

  const int bcx = po_vm["bcx"].as<int>();
  const int bcy = po_vm["bcy"].as<int>();

  // Draw and save the 3x3 operators
  QImage image = draw_init(bcx, bcy, 3, 3);

  draw_base(image, bcx, bcy, 3, 3, false, true);        
  image.save("base.png");

  draw_lbp(image, bcx, bcy);
  image.save("lbp.png");

  draw_dlbp(image, bcx, bcy);
  image.save("dlbp.png");

  draw_tlbp(image, bcx, bcy);
  image.save("tlbp.png");

  draw_mlbp(image, bcx, bcy);
  image.save("mlbp.png");

  draw_mct(image, bcx, bcy);
  image.save("mct.png");

  // Draw and save the 4x2 operators
  image = draw_init(bcx, bcy, 4, 2);

  draw_tlbp4x2(image, bcx, bcy);
  image.save("tlbp4x2.png");

  draw_mlbp4x2(image, bcx, bcy);
  image.save("mlbp4x2.png");

  // Draw and save the 2x4 operators
  image = draw_init(bcx, bcy, 2, 4);

  draw_tlbp2x4(image, bcx, bcy);
  image.save("tlbp2x4.png");

  draw_mlbp2x4(image, bcx, bcy);
  image.save("mlbp2x4.png");

  // Draw and save the 8x1 operators
  image = draw_init(bcx, bcy, 8, 1);

  draw_tlbp8x1(image, bcx, bcy);
  image.save("tlbp8x1.png");

  draw_mlbp8x1(image, bcx, bcy);
  image.save("mlbp8x1.png");

  // Draw and save the 1x8 operators
  image = draw_init(bcx, bcy, 1, 8);

  draw_tlbp1x8(image, bcx, bcy);
  image.save("tlbp1x8.png");

  draw_mlbp1x8(image, bcx, bcy);
  image.save("mlbp1x8.png");

  // OK
  return EXIT_SUCCESS;

}
