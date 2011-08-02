#!/usr/bin/env python
import torch
import train_gmm
import os
from optparse import OptionParser

usage = "usage: %prog [options] "

parser = OptionParser(usage)
parser.set_description("Run a GMM experiment")

parser.add_option("-o",
                  "--output-dir",
                  dest="output_dir",
                  help="Output dir",
                  type="string",
                  default="output")
parser.add_option("-f",
                  "--features-dir",
                  dest="features_dir",
                  default="/scratch/fmoulin/features/final_blockDCT-c12-w12-p11-q11-n45-b-a",
                  type="string",
                  help="Feature dir")
parser.add_option("-d",
                  "--desc",
                  dest="description",
                  type="string",
                  help="",
                  default="")
parser.add_option("-k",
                  "--iterk",
                  dest="iterk",
                  help="",
                  type="int",
                  default=50)
parser.add_option("-g",
                  "--iterg",
                  dest="iterg",
                  help="",
                  type="int",
                  default=50)
parser.add_option("-t",
                  "--var-thd",
                  dest="var_thd",
                  help="",
                  type="float",
                  default=0.001)
parser.add_option("-e",
                  "--end-acc",
                  dest="end_acc",
                  help="",
                  type="float",
                  default=0.0005)
parser.add_option("-n",
                  "--nb-gaussians",
                  dest="nb_gaussians",
                  help="",
                  type="int",
                  default=50)
parser.add_option("-l",
                  "--linear-scoring",
                  dest="linear_scoring",
                  action="store_true",
                  help="",
                  default=False)
parser.add_option("-z",
                  "--ztnorm",
                  dest="ztnorm",
                  action="store_true",
                  help="",
                  default=False)
parser.add_option("-r",
                  "--relevance-factor",
                  dest="relevance_factor",
                  type="float",
                  help="",
                  default="0.001")
parser.add_option("--rt",
                  "--responsibilities_threshold",
                  dest="responsibilities_threshold",
                  type="float",
                  help="",
                  default="0")

(options, args) = parser.parse_args()

protocol='P'
extension='.hdf5'
models_dir = os.path.join(options.output_dir, "models")
if not os.path.exists(models_dir):
  os.mkdir(models_dir)

wm_path = os.path.join(options.output_dir, "wm.hdf5")
db = torch.db.banca.Database()

if os.path.exists(wm_path):
  print "Loading world model"
  wm = torch.machine.GMMMachine(torch.io.HDF5File(wm_path))
else:
  print "Training world model"
  train_files = db.files(directory=options.features_dir, extension=extension,
                         protocol=protocol, purposes='enrol', groups='world', 
                         subworld="twothirds", languages='en')

  data = train_gmm.loadData(train_files.itervalues())

  wm = train_gmm.trainGMM(data, options.nb_gaussians, options.iterk, 
                              options.iterg, options.end_acc, options.var_thd,
                              True, True, True, True)

  wm.save(torch.io.HDF5File(wm_path))
 

exp = train_gmm.GMMExperiment(db, options.features_dir, extension, protocol, wm, models_dir, options.linear_scoring, options.ztnorm)
exp.iterg = options.iterg
exp.iterk = options.iterk
exp.convergence_threshold = options.end_acc
exp.variance_threshold = options.var_thd
exp.relevance_factor = options.relevance_factor
exp.responsibilities_threshold = options.responsibilities_threshold


result_dir=os.path.join(options.output_dir, "results", options.description)
if not os.path.exists(result_dir):
  os.makedirs(result_dir)

print "Dev"
exp.run('dev', open(os.path.join(result_dir, 'scores-dev'), 'w'))

print "Test"
exp.run('eval', open(os.path.join(result_dir, 'scores-test'), 'w'))


