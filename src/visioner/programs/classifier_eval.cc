/**
 * @file visioner/programs/classifier_eval.cc
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

#include "bob/core/logging.h"

#include "bob/visioner/cv/cv_classifier.h"
#include "bob/visioner/util/timer.h"

int main(int argc, char *argv[]) {	

  bob::visioner::CVDetector detector;
  bob::visioner::CVClassifier classifier;

  // Parse the command line
  boost::program_options::options_description po_desc("", 160);
  po_desc.add_options()
    ("help,h", "help message");
  po_desc.add_options()
    ("data", boost::program_options::value<std::string>(), 
     "test datasets");
  detector.add_options(po_desc);
  classifier.add_options(po_desc);

  boost::program_options::variables_map po_vm;
  boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
      .options(po_desc).run(),
      po_vm);
  boost::program_options::notify(po_vm);

  // Check arguments and options
  if (	po_vm.empty() || po_vm.count("help") || 
      !po_vm.count("data") ||
      !detector.decode(po_desc, po_vm) ||
      !classifier.decode(po_desc, po_vm))
  {
    bob::core::error << po_desc << std::endl;
    exit(EXIT_FAILURE);
  }

  const std::string cmd_data = po_vm["data"].as<std::string>();

  // Load the test datasets
  std::vector<std::string> ifiles, gfiles;
  if (bob::visioner::load_listfiles(cmd_data, ifiles, gfiles) == false)
  {
    bob::core::error << "Failed to load the test datasets <" << cmd_data << ">!" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Build the confusion matrix
  bob::visioner::Matrix<uint64_t> hits_mat;
  std::vector<uint64_t> hits_cnt;

  classifier.evaluate(ifiles, gfiles, detector, hits_mat, hits_cnt);

  // Display the confusion matrix
  const uint64_t n_classes = classifier.n_classes();
  const std::vector<std::string>& labels = classifier.param().m_labels;

  const uint64_t str_size = 12;
  const std::string empty_str = bob::visioner::resize(std::string(), str_size);

  const uint64_t sum_hits = std::accumulate(hits_cnt.begin(), hits_cnt.end(), 0);

  // --- header
  bob::core::info << empty_str;
  for (uint64_t c1 = 0; c1 < n_classes; c1 ++)
  {
    bob::core::info << bob::visioner::resize(labels[c1], str_size);
  }
  bob::core::info << bob::visioner::resize("[ERR]", str_size) << std::endl;

  // --- content
  uint64_t sum_hits_c11 = 0;
  for (uint64_t c1 = 0; c1 < n_classes; c1 ++)
  {
    bob::core::info << bob::visioner::resize(labels[c1], str_size);

    for (uint64_t c2 = 0; c2 < n_classes; c2 ++)
    {
      const double c12_dr = 
        100.0 * bob::visioner::inverse(hits_cnt[c1]) * hits_mat(c1, c2);   

      bob::core::info
        << bob::visioner::resize(bob::visioner::round(c12_dr, 2), str_size);
    }

    const double c1_err = 
      100.0 - 100.0 * bob::visioner::inverse(hits_cnt[c1]) * hits_mat(c1, c1);  
    bob::core::info
      << bob::visioner::resize(bob::visioner::round(c1_err, 2), str_size) << std::endl;

    sum_hits_c11 += hits_mat(c1, c1);
  }

  // --- end
  const double err = 
    100.0 - 100.0 * bob::visioner::inverse(sum_hits) * sum_hits_c11;  
  bob::core::info
    << ">>> Average error: " << bob::visioner::round(err, 2) << "%." << std::endl;

  // OK
  bob::core::info << "Program finished successfuly" << std::endl;
  return EXIT_SUCCESS;

}
