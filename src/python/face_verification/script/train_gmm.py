#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Francois Moulin <Francois.Moulin@idiap.ch>
# Tue Aug 2 11:38:01 2011 +0200
#
# Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import bob.io
import bob.machine
import bob.trainer
import os, sys

def NormalizeStdArrayset(arrayset):
  arrayset.load()

  length = arrayset.shape[0]
  n_samples = len(arrayset)
  mean = bob.core.array.float64_1(length)
  std = bob.core.array.float64_1(length)

  mean.fill(0)
  std.fill(0)

  for array in arrayset:
    x = array.get().cast('float64')
    mean += x
    std += (x ** 2)

  mean /= n_samples
  std /= n_samples
  std -= (mean ** 2)
  std = std ** 0.5 # sqrt(std)

  arStd = bob.io.Arrayset()
  for array in arrayset:
    arStd.append(array.get().cast('float64') / std)

  return (arStd,std)


def multiplyVectorsByFactors(matrix, vector):
  for i in range(0, matrix.rows()):
    for j in range(0, matrix.columns()):
      matrix[i, j] *= vector[j]


def loadData(files):
  data = bob.io.Arrayset()
  for f in files:
    data.extend(bob.io.Array(str(f)))

  return data


def trainGMM(data, n_gaussians=5, iterk=25, iterg=25, convergence_threshold=1e-5, variance_threshold=0.001, 
             update_weights=True, update_means=True, update_variances=True, norm_KMeans=False):
  ar = data

  # Compute input size
  input_size = ar.shape[0]

  # Create a normalized sampler
  if not norm_KMeans:
    normalizedAr = ar
  else:
    (normalizedAr,stdAr) = NormalizeStdArrayset(ar)
    
  # Create the machines
  kmeans = bob.machine.KMeansMachine(n_gaussians, input_size)
  gmm = bob.machine.GMMMachine(n_gaussians, input_size)

  # Create the KMeansTrainer
  kmeansTrainer = bob.trainer.KMeansTrainer()
  kmeansTrainer.convergenceThreshold = convergence_threshold
  kmeansTrainer.maxIterations = iterk

  # Train the KMeansTrainer
  kmeansTrainer.train(kmeans, normalizedAr)

  [variances, weights] = kmeans.getVariancesAndWeightsForEachCluster(normalizedAr)
  means = kmeans.means

  # Undo normalization
  if norm_KMeans:
    multiplyVectorsByFactors(means, stdAr)
    multiplyVectorsByFactors(variances, stdAr ** 2)

  # Initialize gmm
  gmm.means = means
  gmm.variances = variances
  gmm.weights = weights
  gmm.setVarianceThresholds(variance_threshold)

  # Train gmm
  trainer = bob.trainer.ML_GMMTrainer(update_means, update_variances, update_weights)
  trainer.convergenceThreshold = convergence_threshold
  trainer.maxIterations = iterg
  trainer.train(gmm, ar)

  return gmm



def adaptGMM(data, prior_gmm, iterg=25, convergence_threshold=1e-5, variance_threshold=0.001, adapt_weight=False, adapt_variance=False, relevance_factor=0.001, responsibilities_threshold=0, bob3_map=False, alpha_bob3=0.5):

  ar=data

  # Load prior gmm
  prior_gmm.setVarianceThresholds(variance_threshold)

  # Create trainer
  if responsibilities_threshold == 0.:
    trainer = bob.trainer.MAP_GMMTrainer(relevance_factor, True, adapt_variance, adapt_weight)
  else:
    trainer = bob.trainer.MAP_GMMTrainer(relevance_factor, True, adapt_variance, adapt_weight, responsibilities_threshold)
  trainer.convergenceThreshold = convergence_threshold
  trainer.maxIterations = iterg
  trainer.setPriorGMM(prior_gmm)

  if bob3_map:
    trainer.setT3MAP(alpha_bob3)

  # Load gmm
  gmm = bob.machine.GMMMachine(prior_gmm)
  gmm.setVarianceThresholds(variance_threshold)

  # Train gmm
  trainer.train(gmm, ar)

  return gmm


class GMMExperiment:
  
  def __init__(self, db, features_dir, extension, protocol, wm, models_dir, linear_scoring=False, ztnorm=False):
    self.features_dir = features_dir
    self.extension = extension
    self.protocol = protocol
    self.wm = wm
    self.db = db
    self.client_models = {}
    self.models_dir = models_dir
    self.linear_scoring = linear_scoring
    self.ztnorm = ztnorm
    self.iterg = 50
    self.convergence_threshold = 1e-5
    self.variance_threshold = 0.001
    self.relevance_factor = 0.001

  def precomputeZTnorm(self, tnorm_clients, znorm_clients):
    print "Loading data for ZTnorm"
    print "Getting models for tnorm_clients: " + str(tnorm_clients)

    i = 0
    self.tnorm_models = []
    for c in tnorm_clients:
      self.tnorm_models.append(self.getModel(c))
      i += 1
      print str(i) + "/" + str(len(tnorm_clients))



    print "Getting statistics for znorm_clients: " + str(znorm_clients)

    tnorm_clients_ext=[]
    i = 0
    self.znorm_tests = []
    for c in znorm_clients:
      train_files = self.db.files(directory=self.features_dir, extension=self.extension, protocol=self.protocol, purposes='probe', model_ids=(c,), groups=None, classes='client', languages='en')
      for f in train_files.itervalues():
        [file_basename, x] = os.path.splitext(os.path.basename(f))
        stat_path =  os.path.join(self.models_dir, "statswm_" + file_basename + "_" + str(c) + ".hdf5")
        if os.path.exists(stat_path):
          stats = bob.machine.GMMStats(bob.io.HDF5File(str(stat_path)))
        else:
          data = loadData([f])
          stats = bob.machine.GMMStats(self.wm.nGaussians, self.wm.nInputs)
          self.wm.accStatistics(data, stats)
          stats.save(bob.io.HDF5File(str(stat_path)))

        self.znorm_tests.append(stats)
        tnorm_clients_ext.append(c)


      i += 1
      print str(i) + "/" + str(len(znorm_clients))

    #print "tnorm_models"
    #print self.tnorm_models[0]
    #print "znorm_tests"
    #print self.znorm_tests[0]
    self.D = bob.machine.linearScoring(self.tnorm_models, self.wm, self.znorm_tests)
    self.D_sameValue = self.sameValue(znorm_clients, tnorm_clients_ext)

    print "Loading data for ZTnorm ... done"
    
  def sameValue(self, vect_A, vect_B):
    sameMatrix = bob.core.array.bool_2(len(vect_A), len(vect_B))

    for j in range(len(vect_A)):
      for i in range(len(vect_B)):
        sameMatrix[j, i] = (vect_A[j] == vect_B[i])

    return sameMatrix

  def setZTnormGroup(self, group):
    if group == 'dev':
      g = 'eval'
    elif group == 'eval':
      g = 'dev'
    else:
      raise Exception("Unknown group " + str(group))
      
    tnorm_clients = self.db.clients(protocol=self.protocol, groups=g, language='en')
    znorm_clients = self.db.clients(protocol=self.protocol, groups=g, language='en')

    self.precomputeZTnorm(tnorm_clients, znorm_clients)
    

  def train(self, model_id):
    print "Training " + str(model_id)
    train_files = self.db.files(directory=self.features_dir, extension=self.extension, protocol=self.protocol, purposes='enrol', model_ids=(model_id,), groups=None, classes='client', languages='en')
    data = loadData(train_files.itervalues())
    gmm = adaptGMM(data, 
                   self.wm, 
                   iterg=self.iterg,
                   convergence_threshold=self.convergence_threshold,
                   variance_threshold=self.variance_threshold,
                   relevance_factor=self.relevance_factor,
                   responsibilities_threshold=self.responsibilities_threshold)
    return gmm
  
  def getModel(self, model_id):
    if not model_id in self.client_models:
      model_path = os.path.join(self.models_dir, str(model_id) + ".hdf5")
      if os.path.exists(model_path):
        self.client_models[model_id] = bob.machine.GMMMachine(bob.io.HDF5File(model_path))
      else:
        self.client_models[model_id] = self.train(model_id)
        self.client_models[model_id].save(bob.io.HDF5File(model_path))
    
    return self.client_models[model_id]
    

  def scores(self, models, files):
    if self.linear_scoring:
      list_stats=[]
      for f in files :
        data = loadData([f])
        stats = bob.machine.GMMStats(self.wm.nGaussians, self.wm.nInputs)
        self.wm.accStatistics(data, stats)
        list_stats.append(stats)
      
      scores = bob.machine.linearScoring(models, self.wm, list_stats)
    else:
      scores = bob.core.array.float64_2(len(models), len(files))
      
      nb_scores = len(models)*len(files)
      i=0
      print str(nb_scores) + " to compute:",
      sys.stdout.flush()
      for m in range(len(models)):
        for f in range(len(files)):
          data = loadData([files[f]])
          sumWm = 0
          sumc = 0
          for d in data:
            sumWm += self.wm.forward(d.get())
            sumc += models[m].forward(d.get())

          scores[m, f] = sumc - sumWm
          i+=1
          print str(i),
          sys.stdout.flush()
      print ""
    
    if self.ztnorm:
      A = scores
      B = bob.machine.linearScoring(models, self.wm, self.znorm_tests)
      C = bob.machine.linearScoring(self.tnorm_models, self.wm, list_stats)
      scores = bob.machine.ztnorm(A, B, C, self.D, self.D_sameValue)
    return scores

  def convert_score_to_list(self, scores, probes):
    ret = []
    i = 0
    for c in probes.itervalues():
      ret.append((c[1], c[2], c[3], c[4], scores[0, i]))
      i+=1

    return ret

  def scores_client(self, model_id):
    client_probes = self.db.objects(directory=self.features_dir, extension=self.extension, protocol=self.protocol, purposes="probe", model_ids=(model_id,), classes="client", languages='en')

    files = [x[0] for x in client_probes.itervalues()]
    scores = self.scores([self.getModel(model_id)], files)
    
    return self.convert_score_to_list(scores, client_probes)

  def scores_impostor(self, model_id):
    client_probes = self.db.objects(directory=self.features_dir, extension=self.extension, protocol=self.protocol, purposes="probe", model_ids=(model_id,), classes="impostor", languages='en')

    files = [x[0] for x in client_probes.itervalues()]
    scores = self.scores([self.getModel(model_id)], files)
    
    return self.convert_score_to_list(scores, client_probes)

  
  def run(self, groups, output_file):
    models = self.db.models(groups=groups)

    if self.ztnorm:
      self.setZTnormGroup(groups)

    i=0
    total=len(models)
    for c in models:
      scores=self.scores_client(c)
      for x in scores:
        output_file.write(str(x[2]) + " " + str(x[0]) + " " + str(x[3]) + " " + str(x[4]) + "\n") 

      scores=self.scores_impostor(c)
      for x in scores:
        output_file.write(str(x[2]) + " " + str(x[0]) + " " + str(x[3]) + " " + str(x[4]) + "\n") 
      
      i+=1
      print str(i) + "/" + str(total)

