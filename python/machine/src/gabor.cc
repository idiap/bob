/**
 * @file python/machine/src/gabor.cc
 * @date 2012-03-05
 * @author Manuel Guenther <Manuel.Guenther@idiap.ch>
 *
 * @brief Bindings for the GaborGraphMachine and several GaborJetSimilarities
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


#include <boost/python.hpp>
#include <core/python/ndarray.h>

#include <ip/GaborWaveletTransform.h>
#include <machine/GaborGraphMachine.h>
#include <machine/GaborJetSimilarities.h>
#include <core/array_exception.h>


static boost::python::tuple bob_disparity(const bob::machine::DisparitySimilarity& self){
  return boost::python::make_tuple(self.disparity()[0], self.disparity()[1]);
}

static void bob_extract(bob::machine::GaborGraphMachine& self, bob::python::const_ndarray input_jet_image, bob::python::ndarray output_graph){
  if (output_graph.type().nd == 2){
    const blitz::Array<double,3> jet_image = input_jet_image.bz<double,3>();
    blitz::Array<double,2> graph = output_graph.bz<double,2>();
    self.extract(jet_image, graph);
  } else if (output_graph.type().nd == 3){
    const blitz::Array<double,4> jet_image = input_jet_image.bz<double,4>();
    blitz::Array<double,3> graph = output_graph.bz<double,3>();
    self.extract(jet_image, graph);
  } else throw bob::core::UnexpectedShapeError();
}

static void bob_average(bob::machine::GaborGraphMachine& self, bob::python::const_ndarray many_graph_jets, bob::python::ndarray averaged_graph_jets){
  const blitz::Array<double,4> graph_set = many_graph_jets.bz<double,4>();
  blitz::Array<double,3> graph = averaged_graph_jets.bz<double,3>();
  self.average(graph_set, graph);
}

static double bob_similarity(bob::machine::GaborGraphMachine& self, bob::python::const_ndarray model_graph, bob::python::ndarray probe_graph, const bob::machine::GaborJetSimilarity& similarity_function){
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
      throw bob::core::UnexpectedShapeError();
  }
}

static double scalar_product_sim(const bob::machine::ScalarProductSimilarity& self, bob::python::const_ndarray jet1, bob::python::const_ndarray jet2){
  switch (jet1.type().nd){
    case 1:{
      const blitz::Array<double,1> j1 = jet1.bz<double,1>(), j2 = jet2.bz<double,1>();
      return self.similarity(j1, j2);
    }
    case 2:{
      const blitz::Array<double,2> j1 = jet1.bz<double,2>(), j2 = jet2.bz<double,2>();
      return self.similarity(j1, j2);
    }
    default:
      throw bob::core::UnexpectedShapeError();
  }
}

static double canberra_sim(const bob::machine::CanberraSimilarity& self, bob::python::const_ndarray jet1, bob::python::const_ndarray jet2){
  switch (jet1.type().nd){
    case 1:{
      const blitz::Array<double,1> j1 = jet1.bz<double,1>(), j2 = jet2.bz<double,1>();
      return self.similarity(j1, j2);
    }
    case 2:{
      const blitz::Array<double,2> j1 = jet1.bz<double,2>(), j2 = jet2.bz<double,2>();
      return self.similarity(j1, j2);
    }
    default:
      throw bob::core::UnexpectedShapeError();
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
      "__call__",
      &bob::machine::GaborJetSimilarity::similarity,
      (boost::python::arg("self"), boost::python::arg("jet1"), boost::python::arg("jet2")),
      "Computes the similarity."
    );

  boost::python::class_<bob::machine::ScalarProductSimilarity, boost::shared_ptr<bob::machine::ScalarProductSimilarity>, boost::python::bases<bob::machine::GaborJetSimilarity> >(
      "ScalarProductSimilarity",
      "This class computes the similarity of two Gabor jets as the normalized scalar product (also known as the cosine measure)"
  )

    .def(
      "__call__",
      &scalar_product_sim,
      (boost::python::arg("self"), boost::python::arg("jet1"), boost::python::arg("jet2")),
      "Computes the similarity."
    );

  boost::python::class_<bob::machine::CanberraSimilarity, boost::shared_ptr<bob::machine::CanberraSimilarity>, boost::python::bases<bob::machine::GaborJetSimilarity> >(
      "CanberraSimilarity",
      "This class computes the similarity of two Gabor jets as the Canberra similarity measure: \\sum_j |a_j - a_j'| / (a_j + a_j'))"
    )

    .def(
      "__call__",
      &canberra_sim,
      (boost::python::arg("self"), boost::python::arg("jet1"), boost::python::arg("jet2")),
      "Computes the similarity."
    );

  boost::python::class_<bob::machine::DisparitySimilarity, boost::shared_ptr<bob::machine::DisparitySimilarity>, boost::python::bases<bob::machine::GaborJetSimilarity> >(
      "DisparitySimilarity",
      "This class computes the similarity of two Gabor jets by computing the disparity between the two jets and use this to correct phase differences in the calculation of the similarity",
      boost::python::no_init
    )

    .def(
      boost::python::init<const bob::ip::GaborWaveletTransform&>(
        (boost::python::arg("gwt") = bob::ip::GaborWaveletTransform()),
        "Initializes the similarity measure with parameters from the given transform (or default transform, if no other is given)"
      )
    )

    .def(
      "__call__",
      &bob::machine::DisparitySimilarity::similarity,
      (boost::python::arg("self"), boost::python::arg("jet1"), boost::python::arg("jet2")),
      "Computes the similarity."
    )

    .def
    (
      "disparity",
      &bob_disparity,
      boost::python::arg("self"),
      "Returns the disparity estimated by the last call to similarity"
    );


  boost::python::class_<bob::machine::DisparityCorrectedPhaseDifference, boost::shared_ptr<bob::machine::DisparityCorrectedPhaseDifference>, boost::python::bases<bob::machine::DisparitySimilarity> >(
      "DisparityCorrectedPhaseDifference",
      "This class computes the similarity of two Gabor jets by computing the disparity between the two jets and use this to correct phase differences in the calculation of the similarity",
      boost::python::no_init
    )

    .def(
      boost::python::init<const bob::ip::GaborWaveletTransform&>(
        (boost::python::arg("gwt") = bob::ip::GaborWaveletTransform()),
        "Initializes the similarity measure with parameters from the given transform (or default transform, if no other is given)"
      )
    )

    .def(
      "__call__",
      &bob::machine::DisparityCorrectedPhaseDifference::similarity,
      (boost::python::arg("self"), boost::python::arg("jet1"), boost::python::arg("jet2")),
      "Computes the similarity."
  );


  boost::python::class_<bob::machine::DisparityCorrectedPhaseDifferencePlusCanberra, boost::shared_ptr<bob::machine::DisparityCorrectedPhaseDifferencePlusCanberra>, boost::python::bases<bob::machine::DisparitySimilarity> >(
      "DisparityCorrectedPhaseDifferencePlusCanberra",
      "This class computes the similarity of two Gabor jets by computing the disparity between the two jets and use this to correct phase differences in the calculation of the similarity. Additionally, the Canberra distance between the absolute values of the jets is added.",
      boost::python::no_init
    )

    .def(
      boost::python::init<const bob::ip::GaborWaveletTransform&>(
        (boost::python::arg("gwt") = bob::ip::GaborWaveletTransform()),
        "Initializes the similarity measure with parameters from the given transform (or default transform, if no other is given)"
      )
    )

    .def(
      "__call__",
      &bob::machine::DisparityCorrectedPhaseDifferencePlusCanberra::similarity,
      (boost::python::arg("self"), boost::python::arg("jet1"), boost::python::arg("jet2")),
      "Computes the similarity."
  );

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
    
    
    .add_property(
      "number_of_nodes",
      &bob::machine::GaborGraphMachine::numberOfNodes,
      "The number of nodes of the graph."
    )
      
    .def(
      "__call__",
      &bob_extract,
      (boost::python::arg("self"), boost::python::arg("jet_image"), boost::python::arg("graph_jets")),
      "Extracts the Gabor jets at the desired locations from the given Gabor jet image"
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
