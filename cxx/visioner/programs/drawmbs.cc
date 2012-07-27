#include <QApplication>
#include <QPainter>
#include <QColor>
#include <boost/program_options.hpp>

#include "visioner/util/util.h"

/**
 * Drawing functions for displaying MB-LBP patterns
 */
QImage draw_init(int bw, int bh, int bcx, int bcy) {
  return QImage(bw * bcx, bh * bcy, QImage::Format_RGB32);
}

void draw_clear(QImage& image, int bw, int bh, int bcx, int bcy) {
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

void draw_mb_mct(QImage& image, int bcx, int bcy, int dx, int dy, int cx, int cy, int nx, int ny)
{        
  // Fill the cells
  QPainter painter(&image);        
  for (int x = 0; x < nx; x ++)
  {
    for (int y = 0; y < ny; y ++)
    {
      painter.fillRect(
          (dx + x * cx) * bcx, (dy + y * cy) * bcy, cx * bcx, cy * bcy, 
          QColor(105, 105, 105, 155));
    }
  }

  // Draw the contour of the cells
  painter.setPen(QPen(QBrush(QColor(55, 55, 55)), 4));
  for (int x = 0; x < nx; x ++)
  {
    for (int y = 0; y < ny; y ++)
    {
      painter.drawRect(
          (dx + x * cx) * bcx, (dy + y * cy) * bcy, cx * bcx, cy * bcy);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{	
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
    bob::visioner::log_error("drawmbs") << po_desc << "\n";
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

  // Draw and save the image
  QImage image = draw_init(bw, bh, bcx, bcy);
  draw_clear(image, bw, bh, bcx, bcy);
  draw_mb_mct(image, bcx, bcy, dx, dy, cx, cy, nx, ny);
  image.save(result.c_str());

  // OK
  return EXIT_SUCCESS;

}
