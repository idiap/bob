/**
 * @file machine/cxx/GaborGraphMachine.cc
 * @date 2012-03-05
 * @author Manuel Guenther <Manuel.Guenther@idiap.ch>
 *
 * @brief Implements the extraction of Gabor graphs
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

#include "bob/machine/GaborGraphMachine.h"
#include <complex>

/**
 * Generates Gabor graph machine that generates grid graphs which will be placed according to the given eye positions
 * @param lefteye  Position of the left eye
 * @param righteye Position of the right eye
 * @param between  Number of nodes to place between the eyes (excluding the eye nodes themselves)
 * @param along    Number of nodes to place left and right of the eye nodes (excluding the eye nodes themselves)
 * @param above    Number of nodes to place above the eyes (excluding the eye nodes themselves)
 * @param below    Number of nodes to place below the eyes (excluding the eye nodes themselves)
 */
bob::machine::GaborGraphMachine::GaborGraphMachine(
  blitz::TinyVector<int,2> lefteye,
  blitz::TinyVector<int,2> righteye,
  int between,
  int along,
  int above,
  int below
)
{
  // shortcuts for eye positions
  int lex = lefteye[1], ley = lefteye[0];
  int rex = righteye[1], rey = righteye[0];
  // compute grid parameters
  double stepx = double(lex - rex) / double(between+1);
  double stepy = double(ley - rey) / double(between+1);
  double xstart = rex - along*stepx + above*stepy;
  double ystart = rey - along*stepy - above*stepx;
  int xcount = between + 2 * (along+1);
  int ycount = above + below + 1;

  // create grid positions
  m_node_positions.resize(xcount*ycount,2);
  for (int y = 0, i = 0; y < ycount; ++y){
    for (int x = 0; x < xcount; ++x, ++i){
      // y position
      m_node_positions(i,0) = round(ystart + y * stepx + x * stepy);
      // x position
      m_node_positions(i,1) = round(xstart + x * stepx - y * stepy);
    }
  }
}

/**
 * Generates a Gabor graph machine that will produce regular grid graphs
 * starting at the given first index, ending at (or before) the given last index,
 * and advancing the given step size.
 * @param first  First node to be placed (top-left)
 * @param last   Last node to be placed (bottom-right). Depending on the step size, this node might not be reached.
 * @param step   The step size (in pixel) between two nodes
 */
bob::machine::GaborGraphMachine::GaborGraphMachine(
  blitz::TinyVector<int,2> first,
  blitz::TinyVector<int,2> last,
  blitz::TinyVector<int,2> step
)
{
  int ycount = (last[0] - first[0]) / step[0] + 1;
  int xcount = (last[1] - first[1]) / step[1] + 1;

  // create grid positions
  m_node_positions.resize(xcount*ycount,2);
  for (int y = 0, i = 0; y < ycount; ++y){
    for (int x = 0; x < xcount; ++x, ++i){
      // y position
      m_node_positions(i,0) = first[0] + y * step[0];
      // x position
      m_node_positions(i,1) = first[1] + x * step[1];
    }
  }
}

/**
 * Generates this machine as a copy of the other one
 *
 * @param other  The machine to copy
 */
bob::machine::GaborGraphMachine::GaborGraphMachine(
  const GaborGraphMachine& other
)
{
  m_node_positions.resize(other.m_node_positions.shape());
  m_node_positions = other.m_node_positions;
}


/**
 * Makes this machine a deep copy of the given one
 *
 * @param other  The machine to copy
 * @return  A reference to *this
 */
bob::machine::GaborGraphMachine& bob::machine::GaborGraphMachine::operator =(
  const GaborGraphMachine& other
)
{
  m_node_positions.resize(other.m_node_positions.shape());
  m_node_positions = other.m_node_positions;
  return *this;
}

/**
 * Checks if the parameterization of both machines is identical.
 *
 * @param other  The machine to test for equality to this
 * @return true if the node positions of both machines are identical, otherwise false
 */
bool bob::machine::GaborGraphMachine::operator ==(
  const GaborGraphMachine& other
) const
{
  return (blitz::all(m_node_positions == other.m_node_positions));
}


void bob::machine::GaborGraphMachine::checkPositions(int height, int width) const throw(){
  for (int i = m_node_positions.extent(0); i--;){
    if (m_node_positions(i,0) < 0 || m_node_positions(i,0) >= height ||
        m_node_positions(i,1) < 0 || m_node_positions(i,1) >= width)
      throw ImageTooSmallException(height, width, m_node_positions(i,0), m_node_positions(i,1));
  }
}

/**
 * Extracts the Gabor jets (including phase information) at the node positions
 * @param jet_image  The Gabor jet image to extract the Gabor jets from
 * @param graph_jets The graph that will be filled
 */
void bob::machine::GaborGraphMachine::extract(
  const blitz::Array<double,4>& jet_image,
  blitz::Array<double,3>& graph_jets
) const {
  // check the positions
  checkPositions(jet_image.shape()[0], jet_image.shape()[1]);
  // extract Gabor jets
  blitz::Range all = blitz::Range::all();
  for (int i = 0; i < m_node_positions.extent(0); ++i){
    graph_jets(i,all,all) = jet_image(m_node_positions(i,0), m_node_positions(i,1), all, all);
  }
}

/**
 * Extracts the Gabor jets (without phase information) at the node positions
 * @param jet_image  The Gabor jet image to extract the Gabor jets from
 * @param graph_jets The graph that will be filled
 */
void bob::machine::GaborGraphMachine::extract(
  const blitz::Array<double,3>& jet_image,
  blitz::Array<double,2>& graph_jets
) const {
  // check the positions
  checkPositions(jet_image.shape()[0], jet_image.shape()[1]);
  // extract Gabor jets
  blitz::Range all = blitz::Range::all();
  for (int i = 0; i < m_node_positions.extent(0); ++i){
    graph_jets(i,all) = jet_image(m_node_positions(i,0), m_node_positions(i,1), all);
  }
}


/**
 * Averages the given set of Gabor graphs into a single one by interpolating the Gabor jets
 * @param many_graph_jets     The set of Gabor graphs to average
 * @param averaged_graph_jets The averaged Gabor graph
 */
void bob::machine::GaborGraphMachine::average(
  const blitz::Array<double,4>& many_graph_jets,
  blitz::Array<double,3>& averaged_graph_jets
) const {

  m_averages.resize(many_graph_jets.extent(1));

  // iterate over the Gabor jets
  for (int i = 0; i < many_graph_jets.extent(1); ++i){
    m_averages = std::complex<double>(0.);
    for (int p = 0; p < many_graph_jets.extent(0); ++p){
      for (int j = 0; j < many_graph_jets.extent(3); ++j){
        // totalize Gabor jet entries as complex numbers
        m_averages(j) += std::polar(many_graph_jets(p,i,0,j), many_graph_jets(p,i,1,j));
      }
    }

    blitz::Array<double,2> current_jet(averaged_graph_jets(i, blitz::Range::all(), blitz::Range::all()));
    // now, copy the data back to the polar description
    for (int j = 0; j < many_graph_jets.extent(3); ++j){
      current_jet(0,j) = std::abs(m_averages(j));
      current_jet(1,j) = std::arg(m_averages(j));
    }

    // normalize the length of the Gabor jet
    bob::ip::normalizeGaborJet(current_jet);
  }
}


/**
 * Computes the similarity of two Gabor graphs with the given similarity function
 * @param model_graph_jets  One of the two graphs to compare
 * @param probe_graph_jets  One of the two graphs to compare
 * @param jet_similarity_function  The similarity function to be used for comparison of two corresponding Gabor jets
 * @return the similarity of the two graphs
 */
double bob::machine::GaborGraphMachine::similarity(
  const blitz::Array<double,2>& model_graph_jets,
  const blitz::Array<double,2>& probe_graph_jets,
  const bob::machine::GaborJetSimilarity& jet_similarity_function
) const
{
  // iterate over the nodes and average Gabor jet similarities
  double similarity = 0.;
  blitz::Range all = blitz::Range::all();
  for (int i = 0; i < model_graph_jets.extent(0); ++i){
    similarity += jet_similarity_function(model_graph_jets(i,all), probe_graph_jets(i,all));
  }
  return similarity / model_graph_jets.extent(0);
}

/**
 * Computes the similarity of two Gabor graphs with the given similarity function
 * @param model_graph_jets  One of the two graphs to compare
 * @param probe_graph_jets  One of the two graphs to compare
 * @param jet_similarity_function  The similarity function to be used for comparison of two corresponding Gabor jets
 * @return the similarity of the two graphs
 */
double bob::machine::GaborGraphMachine::similarity(
  const blitz::Array<double,3>& model_graph_jets,
  const blitz::Array<double,3>& probe_graph_jets,
  const bob::machine::GaborJetSimilarity& jet_similarity_function
) const
{
  // iterate over the nodes and average Gabor jet similarities
  double similarity = 0.;
  blitz::Range all = blitz::Range::all();
  for (int i = 0; i < model_graph_jets.extent(0); ++i){
    similarity += jet_similarity_function(model_graph_jets(i,all,all), probe_graph_jets(i,all,all));
  }
  return similarity / model_graph_jets.extent(0);
}


/**
 * Computes the similarity of the given set of graphs and the given probe graph
 * @param many_model_graph_jets  The set of Gabor graphs to compare
 * @param probe_graph_jets  The probe graph to compare
 * @param jet_similarity_function  The similarity function to be used for comparison of two corresponding Gabor jets
 * @return the similarity of the two graphs
 */
double bob::machine::GaborGraphMachine::similarity(
  const blitz::Array<double,3>& many_model_graph_jets,
  const blitz::Array<double,2>& probe_graph_jets,
  const bob::machine::GaborJetSimilarity& jet_similarity_function
) const
{
  // iterate over the nodes and average Gabor jet similarities
  double similarity = 0.;
  blitz::Range all = blitz::Range::all();
  for (int i = 0; i < many_model_graph_jets.extent(1); ++i){
    // maximize jet similarity over all models in the gallery
    double max_similarity = 0.;
    for (int p = 0; p < many_model_graph_jets.extent(0); ++p){
      max_similarity = std::max(max_similarity, jet_similarity_function(many_model_graph_jets(p,i,all), probe_graph_jets(i,all)));
    }
    similarity += max_similarity;
  }
  return similarity / many_model_graph_jets.extent(1);
}


/**
 * Computes the similarity of the given set of graphs and the given probe graph
 * @param many_model_graph_jets  The set of Gabor graphs to compare
 * @param probe_graph_jets  The probe graph to compare
 * @param jet_similarity_function  The similarity function to be used for comparison of two corresponding Gabor jets
 * @return the similarity of the two graphs
 */
double bob::machine::GaborGraphMachine::similarity(
  const blitz::Array<double,4>& many_model_graph_jets,
  const blitz::Array<double,3>& probe_graph_jets,
  const bob::machine::GaborJetSimilarity& jet_similarity_function
) const
{
  // iterate over the nodes and average Gabor jet similarities
  double similarity = 0.;
  blitz::Range all = blitz::Range::all();
  for (int i = 0; i < many_model_graph_jets.extent(1); ++i){
    // maximize jet similarity over all models in the gallery
    double max_similarity = 0.;
    for (int p = 0; p < many_model_graph_jets.extent(0); ++p){
      max_similarity = std::max(max_similarity, jet_similarity_function(many_model_graph_jets(p,i,all,all), probe_graph_jets(i,all,all)));
    }
    similarity += max_similarity;
  }
  return similarity / many_model_graph_jets.extent(1);
}


void bob::machine::GaborGraphMachine::save(bob::io::HDF5File& file) const{
  file.setArray("NodePositions", m_node_positions);
}

void bob::machine::GaborGraphMachine::load(bob::io::HDF5File& file){
  m_node_positions = file.readArray<int,2>("NodePositions");
}
