#include <QApplication>
#include <QPainter>
#include <QColor>
#include <boost/program_options.hpp>

#include "visioner/util/util.h"

QImage draw_init(int bw, int bh, int bcx, int bcy)
{
  return QImage(bw * bcx, bh * bcy, QImage::Format_RGB32);
}

void draw_clear(QImage& image, int bw, int bh, int bcx, int bcy)
{
  // Draw the background
  QPainter painter(&image);
  painter.fillRect(painter.window(), QColor(255, 255, 255));

  painter.setPen(QPen(QBrush(QColor(175, 175, 175)), 2));
  for (int x = 0; x < bw; x ++)
  {
    for (int y = 0; y < bh; y ++)
    {
      painter.drawRect(x * bcx, y * bcy, bcx, bcy);
    }
  }
}

void draw_cells(QImage& image, int bcx, int bcy, int dx, int dy, int cx, int cy, int nx, int ny)
{
  QPainter painter(&image);

  // Draw the cells
  for (int x = 0; x < nx; x ++)
  {
    for (int y = 0; y < ny; y ++)
    {
      painter.fillRect(
          (dx + x * cx) * bcx, (dy + y * cy) * bcy, cx * bcx, cy * bcy, 
          QColor(105, 105, 105, 155));
    }
  }
}

void draw_contour(QImage& image, int bcx, int bcy, int dx, int dy, int cx, int cy, int nx, int ny,
    const QPen& pen = QPen(QBrush(QColor(55, 55, 55)), 4, Qt::SolidLine))
{
  QPainter painter(&image);

  // Draw the contour of the cells
  painter.setPen(pen);
  for (int x = 0; x < nx; x ++)
  {
    for (int y = 0; y < ny; y ++)
    {
      const int t = (dy + y * cy) * bcy, l = (dx + x * cx) * bcx;
      const int w = cx * bcx, h = cy * bcy;

      painter.drawLine(l, t, l + w, t);
      painter.drawLine(l, t, l, t + h);
      painter.drawLine(l + w, t, l + w, t + h);
      painter.drawLine(l, t + h, l + w, t + h);
    }
  }
}

void draw_center(QImage& image, int bcx, int bcy, int dx, int dy, int cx, int cy, int nx, int ny)
{
  QPainter painter(&image);

  // Draw the center of the feature
  painter.setPen(QPen(QBrush(QColor(255, 55, 55)), 4));
  const int centerx = (dx + nx * cx / 2) * bcx;
  const int centery = (dy + ny * cy / 2) * bcy;
  painter.drawLine(centerx - bcx / 2, centery, centerx + bcx / 2, centery);
  painter.drawLine(centerx, centery - bcy / 2, centerx, centery + bcy / 2);
}

void draw_mb_mct(QImage& image, int bcx, int bcy, int dx, int dy, int cx, int cy, int nx, int ny,
    const QPen& pen_contour = QPen(QBrush(QColor(55, 55, 55)), 4, Qt::SolidLine))
{        
  draw_cells(image, bcx, bcy, dx, dy, cx, cy, nx, ny);
  draw_contour(image, bcx, bcy, dx, dy, cx, cy, nx, ny, pen_contour);
  draw_center(image, bcx, bcy, dx, dy, cx, cy, nx, ny);
}

//QImage draw_combine(const QImage& orig_image, const QImage* proj_images)
//{
//        // Create the combined image
//        const int bx = orig_image.width() / 2;
//        const int by = orig_image.height() / 2;
//        const QImage& proj_image = proj_images[0];
//        const int w = proj_image.width() * 4;
//        const int h = proj_image.height() * 4;

//        QImage result(w, h, QImage::Format_RGB32);

//        // Background
//        QPainter painter(&result);        
//        painter.fillRect(painter.window(), qRgb(255, 255, 255));        

//        // Draw the original feature
//        painter.drawImage((w - orig_image.width()) / 2,
//                          (h - orig_image.height()) / 2,
//                          orig_image);        

//        // Draw and connect each projected feature
//        for (int i = 0; i < 9; i ++)
//        {
//                const QImage& proj_image = proj_images[i];
//                const int x = w / 2 + (int)(cos(2.0 * M_PI / 9.0 * i) * radius + 0.5);
//                const int y = h / 2 + (int)(sin(2.0 * M_PI / 9.0 * i) * radius + 0.5);

//                painter.drawImage(x - proj_image.width() / 2,
//                                  y - proj_image.height() / 2,
//                                  proj_image);

//                const QPen pen_line(QColor(75, 75, 75), orig_image.width() / 48, Qt::DotLine);
//                painter.setPen(pen_line);

//                const double arrow_angle = M_PI / 8.0;
//                const double arrow_size = orig_image.width() / 24;

//                // Draw the connection line
//                const int startx = w / 2, starty = h / 2;
//                const int stopx = x, stopy = y;
//                painter.drawLine(startx, starty, stopx, stopy);

//                // Draw the arrow at the end of the connection line
//                //
//                // HowTo:	compute the arrow's points as the line was vertical and then multiply
//                //		with the rotation matrix of the line facing axes
//                //
//                const double angle = M_PI * 0.5 - std::atan2((double)(stopy - starty), (double)(stopx - startx));

//                QPoint arrow[3];
//                arrow[0].setX(stopx);
//                arrow[0].setY(stopy);

//                arrow[1].setX((int)(0.5 + stopx + arrow_size * sin(arrow_angle - angle)));
//                arrow[1].setY((int)(0.5 + stopy - arrow_size * cos(arrow_angle - angle)));

//                arrow[2].setX((int)(0.5 + stopx - arrow_size * sin(arrow_angle + angle)));
//                arrow[2].setY((int)(0.5 + stopy - arrow_size * cos(arrow_angle + angle)));

//                painter.drawPolygon(arrow, 3);

//        }

//        return result;
//}

int main(int argc, char *argv[]) {	

  QApplication app(argc, argv);
  Q_UNUSED(app);

  // Parse the command line
  boost::program_options::options_description po_desc("", 160);
  po_desc.add_options()
    ("help,h", "help message");
  po_desc.add_options()
    ("bw", boost::program_options::value<int>()->default_value(16),
     "board: width in cells")
    ("bh", boost::program_options::value<int>()->default_value(16),
     "board: height in cells")
    ("bcx", boost::program_options::value<int>()->default_value(12),
     "board: cell width in pixels")
    ("bcy", boost::program_options::value<int>()->default_value(12),
     "board: cell height in pixels")
    ("dx", boost::program_options::value<int>()->default_value(2),
     "pattern: translation in Ox direction")
    ("dy", boost::program_options::value<int>()->default_value(2),
     "pattern: translation in Oy direction")
    ("cx", boost::program_options::value<int>()->default_value(4),
     "pattern: cell width")
    ("cy", boost::program_options::value<int>()->default_value(4),
     "pattern: cell height")
    ("nx", boost::program_options::value<int>()->default_value(3),
     "pattern: number of cells in Ox direction")
    ("ny", boost::program_options::value<int>()->default_value(3),
     "pattern: number of cells in Oy direction")
    ("result", boost::program_options::value<std::string>()->default_value("result.png"),
     "filename to save the image");

  boost::program_options::variables_map po_vm;
  boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
      .options(po_desc).run(),
      po_vm);
  boost::program_options::notify(po_vm);

  // Check arguments and options
  if (	po_vm.empty() || po_vm.count("help"))
  {
    visioner::log_error("drawmb_ctf") << po_desc << "\n";
    exit(EXIT_FAILURE);
  }

  const int bw = po_vm["bw"].as<int>();
  const int bh = po_vm["bh"].as<int>();
  const int bcx = po_vm["bcx"].as<int>();
  const int bcy = po_vm["bcy"].as<int>();

  const int dx = po_vm["dx"].as<int>();        
  const int dy = po_vm["dy"].as<int>();
  const int cx = po_vm["cx"].as<int>();
  const int cy = po_vm["cy"].as<int>();
  const int nx = po_vm["nx"].as<int>();
  const int ny = po_vm["ny"].as<int>();

  const std::string result = po_vm["result"].as<std::string>();

  // Draw the coarse multi-block pattern and save the image
  QImage image = draw_init(bw, bh, bcx, bcy);
  draw_clear(image, bw, bh, bcx, bcy);
  draw_mb_mct(image, bcx, bcy, dx, dy, cx, cy, nx, ny);
  image.save(result.c_str());

  // Draw the fine (projected) multi-block patterns and save the images
  QImage proj_images[9];
  const int centerx = 2 * dx + nx * cx, centery = 2 * dy + ny * cy;
  for (int proj_cx = 2 * cx - 2, cnt = 0; proj_cx <= 2 * cx + 2; proj_cx += 2)
  {
    for (int proj_cy = 2 * cy - 2; proj_cy <= 2 * cy + 2; proj_cy += 2, cnt ++)
    {
      proj_images[cnt] = draw_init(2 * bw, 2 * bh, bcx, bcy);

      QImage& proj_image = proj_images[cnt];
      draw_clear(proj_image, 2 * bw, 2 * bh, bcx, bcy);                        

      draw_mb_mct(proj_image, bcx, bcy, centerx - proj_cx * nx / 2, centery - proj_cy * ny / 2,
          proj_cx, proj_cy, nx, ny, 
          QPen(QBrush(QColor(55, 55, 55)), 4, Qt::SolidLine));
      draw_contour(proj_image, bcx, bcy, 2 * dx, 2 * dy, 2 * cx, 2 * cy, nx, ny,
          QPen(QBrush(QColor(155, 55, 55)), 4, Qt::DashLine));                        

      proj_image.save((visioner::basename(result) + ".proj" + 
            boost::lexical_cast<visioner::string_t>(cnt) +
            visioner::extname(result)).c_str());
    }
  }

  //        // Assembly the original feature and the projected features
  //        QImage comb_image = draw_combine(image, proj_images);
  //        comb_image.save((visioner::basename(result) + ".comb" +
  //                         visioner::extname(result)).c_str());

  // OK
  return EXIT_SUCCESS;

}
