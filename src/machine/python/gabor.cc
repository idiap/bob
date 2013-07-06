/**
 * @file machine/python/gabor.cc
 * @date 2012-03-05
 * @author Manuel Guenther <Manuel.Guenther@idiap.ch>
 *
 * @brief Bindings for the GaborGraphMachine and several GaborJetSimilarities
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


#include <boost/python.hpp>
#include <bob/python/ndarray.h>

#include <bob/ip/GaborWaveletTransform.h>
#include <bob/machine/GaborGraphMachine.h>
#include <bob/machine/GaborJetSimilarities.h>
#include <bob/core/array_exception.h>


static void bob_extract(bob::machine::GaborGraphMachine& self, bob::python::const_ndarray input_jet_image, bob::python::ndarray output_graph){
  if (output_graph.type().nd == 2){
    const blitz::Array<double,3> jet_image = input_jet_image.bz<double,3>();
    blitz::Array<double,2> graph = output_graph.bz<double,2>();
    self.extract(jet_image, graph);
  } else if (output_graph.type().nd == 3){
    const blitz::Array<double,4> jet_image = input_jet_image.bz<double,4>();
    blitz::Array<double,3> graph = output_graph.bz<double,3>();
    self.extract(jet_image, graph);
  } else {
    throw bob::core::array::UnexpectedShapeError();
  }
}

static bob::python::ndarray bob_extract2(bob::machine::GaborGraphMachine& self, bob::python::const_ndarray input_jet_image){
  if (input_jet_image.type().nd == 3){
    const blitz::Array<double,3> jet_image = input_jet_image.bz<double,3>();
    bob::python::ndarray output_graph(bob::core::array::t_float64, self.numberOfNodes(), jet_image.shape()[2]);
    blitz::Array<double,2> graph = output_graph.bz<double,2>();
    self.extract(jet_image, graph);
    return output_graph;
  } else if (input_jet_image.type().nd == 4){
    const blitz::Array<double,4> jet_image = input_jet_image.bz<double,4>();
    bob::python::ndarray output_graph(bob::core::array::t_float64, self.numberOfNodes(), jet_image.shape()[2], jet_image.shape()[3]);
    blitz::Array<double,3> graph = output_graph.bz<double,3>();
    self.extract(jet_image, graph);
    return output_graph;
  } else throw bob::core::array::UnexpectedShapeError();
}

static void bob_average(bob::machine::GaborGraphMachine& self, bob::python::const_ndarray many_graph_jets, bob::python::ndarray averaged_graph_jets){
  const blitz::Array<double,4> graph_set = many_graph_jets.bz<double,4>();
  blitz::Array<double,3> graph = averaged_graph_jets.bz<double,3>();
  self.average(graph_set, graph);
}

static double bob_similarity(bob::machine::GaborGraphMachine& self, bob::python::const_ndarray model_graph, bob::python::ndarray probe_graph, const bob::machine::GaborJetSimilarity& similarity_function){
  switch (probe_graph.type().nd){
    case 2:{ // Gabor graph including jets without phases
      blitz::Array<double,2> probe = probe_graph.bz<double,2>();
      switch (model_graph.type().nd){
        case 2:{
          const blitz::Array<double,2> model = model_graph.bz<double,2>();
          return self.similarity(model, probe, similarity_function);
        }
        case 3:{
          const blitz::Array<double,3> model = model_graph.bz<double,3>();
          return self.similarity(model, probe, similarity_function);
        }
        default:
          throw bob::core::array::UnexpectedShapeError();
      }
    }

    case 3:{ // Gabor graph including jets with phases
      blitz::Array<double,3> probe = probe_graph.bz<double,3>();
      switch (model_graph.type().nd){
        case 3:{
          const blitz::Array<double,3> model = model_graph.bz<double,3>();
          return self.similarity(model, probe, similarity_function);
        }
        case 4:{
          const blitz::Array<double,4> model = model_graph.bz<double,4>();
          return self.similarity(model, probe, similarity_function);
        }
        default:
          throw bob::core::array::UnexpectedShapeError();
      }
    }

    default: // unknown graph shape
      throw bob::core::array::UnexpectedShapeError();
  }
}

static double bob_jet_sim(const bob::machine::GaborJetSimilarity& self, bob::python::const_ndarray jet1, bob::python::const_ndarray jet2){
  switch (jet1.type().nd){
    case 1:{
      const blitz::Array<double,1> j1 = jet1.bz<double,1>(), j2 = jet2.bz<double,1>();
      return self(j1, j2);
    }
    case 2:{
      const blitz::Array<double,2> j1 = jet1.bz<double,2>(), j2 = jet2.bz<double,2>();
      return self(j1, j2);
    }
    default:
      throw bob::core::array::UnexpectedShapeError();
  }
}

void bind_machine_gabor(){
  /////////////////////////////////////////////////////////////////////////////////////////
  //////////////// Gabor jet similarities
  boost::python::class_<bob::machine::GaborJetSimilarity, boost::noncopyable >(
      "GaborJetSimilarity",
      "This is the pure virtual base class for all Gabor jet similarities.",
      boost::python::no_init
    )

    .def(
      boost::python::init<bob::machine::GaborJetSimilarity::SimilarityType, const bob::ip::GaborWaveletTransform&>(
        (
          boost::python::arg("type"),
          boost::python::arg("gwt") = bob::ip::GaborWaveletTransform()
        ),
        "Generates a Gabor jet similarity measure of the given type. The parameters of the given transform are used for disparity-like similarity functions only."
      )
    )

    .def(
      "save",
      &bob::machine::GaborJetSimilarity::save,
      "Saves the parameterization of this Gabor jet similarity function to HDF5 file."
    )

    .def(
      "load",
      &bob::machine::GaborJetSimilarity::load,
      "Loads the parameterization of this Gabor jet similarity function from HDF5 file."
    )

    .def(
      "disparity",
      &bob::machine::GaborJetSimilarity::disparity,
      "Returns the disparity computed by the latest call. Only valid for disparity-like similarity function types."
    )

    .def(
      "__call__",
      &bob_jet_sim,
      (boost::python::arg("self"), boost::python::arg("jet1"), boost::python::arg("jet2")),
      "Computes the similarity between the given Gabor jets."
  );

  boost::python::enum_<bob::machine::GaborJetSimilarity::SimilarityType>("gabor_jet_similarity_type")
    .value("SCALAR_PRODUCT", bob::machine::GaborJetSimilarity::SCALAR_PRODUCT)
    .value("CANBERRA", bob::machine::GaborJetSimilarity::CANBERRA)
    .value("DISPARITY", bob::machine::GaborJetSimilarity::DISPARITY)
    .value("PHASE_DIFF", bob::machine::GaborJetSimilarity::PHASE_DIFF)
    .value("PHASE_DIFF_PLUS_CANBERRA", bob::machine::GaborJetSimilarity::PHASE_DIFF_PLUS_CANBERRA)
    .export_values();

  /////////////////////////////////////////////////////////////////////////////////////////
  //////////////// Gabor graph machine
  boost::python::class_<bob::machine::GaborGraphMachine, boost::shared_ptr<bob::machine::GaborGraphMachine> >(
      "GaborGraphMachine",
      "This class implements functionality dealing with Gabor graphs, Gabor graph comparison and Gabor graph averaging.",
      boost::python::no_init
    )

    .def(
      boost::python::init<>(
        boost::python::arg("self"),
        "Generates an empty Grid graph extractor. This extractor should only be used to compute average graphs or to compare two graphs!"
      )
    )

    .def(
      boost::python::init<const bob::machine::GaborGraphMachine&>(
          "Constructs a GaborGraphMachine from the one by doing a deep copy."
      )
    )

    .def(
      boost::python::self == boost::python::self
    )

    .def(
      boost::python::init<blitz::TinyVector<int,2>, blitz::TinyVector<int,2>, int, int, int, int>(
        (boost::python::arg("self"), boost::python::arg("lefteye"), boost::python::arg("righteye"), boost::python::arg("between")=3, boost::python::arg("along")=2, boost::python::arg("above")=4, boost::python::arg("below")=6),
        "Generates a Grid graph extractor with nodes put according to the given eye positions, and the given number of nodes between, along, above, and below the eyes."
      )
    )

    .def(
      boost::python::init<blitz::TinyVector<int,2>, blitz::TinyVector<int,2>, blitz::TinyVector<int,2> >(
        (boost::python::arg("self"), boost::python::arg("first"), boost::python::arg("last"), boost::python::arg("step")),
        "Generates a Grid graph extractor with nodes put between the given first and last position in the desired step size."
      )
    )

    .def(
      "save",
      &bob::machine::GaborGraphMachine::save,
      "Saves the parameterization of this Gabor graph extractor to HDF5 file."
    )

    .def(
      "load",
      &bob::machine::GaborGraphMachine::load,
      "Loads the parameterization of this Gabor graph extractor from HDF5 file."
    )

    .add_property(
      "number_of_nodes",
      &bob::machine::GaborGraphMachine::numberOfNodes,
      "The number of nodes of the graph."
    )

    .add_property(
      "nodes",
      &bob::machine::GaborGraphMachine::nodes,
      "The node positions of the graph."
      )

    .def(
      "__call__",
      &bob_extract,
      (boost::python::arg("self"), boost::python::arg("jet_image"), boost::python::arg("graph_jets")),
      "Extracts the Gabor jets at the desired locations from the given Gabor jet image"
    )

    .def(
      "__call__",
      &bob_extract2,
      (boost::python::arg("self"), boost::python::arg("jet_image")),
      "Extracts and returns the Gabor jets at the desired locations from the given Gabor jet image"
    )

    .def(
      "average",
      &bob_average,
      (boost::python::arg("self"), boost::python::arg("many_graph_jets"), boost::python::arg("averaged_graph_jets")),
      "Averages the given list of Gabor graphs into one Gabor graph"
    )

    .def(
      "similarity",
      &bob_similarity,
      (boost::python::arg("self"), boost::python::arg("model_graph_jets"), boost::python::arg("probe_graph_jets"), boost::python::arg("jet_similarity_function")),
      "Computes the similarity between the given probe graph and the gallery, which might be a single graph or a collection of graphs"
  );

}
