#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
# Mon Aug 8 20:20:16 2011 +0200
#
# Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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

"""Test trainer package
"""
import os, sys, tempfile, shutil, math
import unittest
import bob
import numpy

def normalize_blocks(src):
  for i in range(src.shape[0]):
    block = src[i, :, :]
    mean = block.mean()
    std = ((block - mean) ** 2).sum() / block.size
    if std == 0:
      std = 1
    else:
      std = math.sqrt(std)

    src[i, :, :] = (block - mean) / std
    
def normalize_dct(src):
  for i in range(src.shape[1]):
    col = src[:, i]
    mean = col.mean()
    std = ((col - mean) ** 2).sum() / col.size
    if std == 0:
      std = 1
    else:
      std = math.sqrt(std)

    src[:, i] = (col - mean) / std


def dctfeatures(prep, A_BLOCK_H, A_BLOCK_W, A_OVERLAP_H, A_OVERLAP_W, 
    A_N_DCT_COEF, norm_before, norm_after, add_xy):
  
  blockShape = bob.ip.get_block_shape(prep, A_BLOCK_H, A_BLOCK_W, A_OVERLAP_H, A_OVERLAP_W)
  blocks = numpy.ndarray(blockShape, 'float64')
  bob.ip.block(prep, blocks, A_BLOCK_H, A_BLOCK_W, A_OVERLAP_H, A_OVERLAP_W)

  if norm_before:
    normalize_blocks(blocks)

  if add_xy:
    real_DCT_coef = A_N_DCT_COEF - 2
  else:
    real_DCT_coef = A_N_DCT_COEF

  
  # Initialize cropper and destination array
  DCTF = bob.ip.DCTFeatures(A_BLOCK_H, A_BLOCK_W, A_OVERLAP_H, A_OVERLAP_W, real_DCT_coef)
  
  # Call the preprocessing algorithm
  dct_blocks = DCTF(blocks)

  n_blocks = blockShape[0]

  dct_blocks_min = 0
  dct_blocks_max = A_N_DCT_COEF
  TMP_tensor_min = 0
  TMP_tensor_max = A_N_DCT_COEF

  if norm_before:
    dct_blocks_min += 1
    TMP_tensor_max -= 1

  if add_xy:
    dct_blocks_max -= 2
    TMP_tensor_min += 2
  
  TMP_tensor = numpy.ndarray((n_blocks, TMP_tensor_max), 'float64')
  
  nBlocks = bob.ip.get_n_blocks(prep, A_BLOCK_H, A_BLOCK_W, A_OVERLAP_H, A_OVERLAP_W)
  for by in range(nBlocks[0]):
    for bx in range(nBlocks[1]):
      bi = bx + by * nBlocks[1]
      if add_xy:
        TMP_tensor[bi, 0] = bx
        TMP_tensor[bi, 1] = by
      
      TMP_tensor[bi, TMP_tensor_min:TMP_tensor_max] = dct_blocks[bi, dct_blocks_min:dct_blocks_max]

  if norm_after:
    normalize_dct(TMP_tensor)

  return TMP_tensor

def face_normalized(img_input, pos_input, features_output):
  # Parameters
  # Cropping
  CROP_EYES_D = 33
  CROP_H = 80
  CROP_W = 64
  CROP_OH = 16
  CROP_OW = 32
      
  # Tan Triggs
  GAMMA = 0.2 
  SIGMA0 = 1.
  SIGMA1 = 2.
  SIZE = 5
  THRESHOLD = 10.
  ALPHA = 0.1

  # DCT blocks
  BLOCK_H = 8
  BLOCK_W = 8
  OVERLAP_H = 7
  OVERLAP_W = 7
  N_DCT_COEF = 15

  # Initialize cropper and destination array
  FEN = bob.ip.FaceEyesNorm( CROP_EYES_D, CROP_H, CROP_W, CROP_OH, CROP_OW)
  cropped_img = numpy.ndarray((CROP_H, CROP_W), 'float64')

  # Initialize the Tan and Triggs preprocessing
  TT = bob.ip.TanTriggs( GAMMA, SIGMA0, SIGMA1, SIZE, THRESHOLD, ALPHA)
  preprocessed_img = numpy.ndarray((CROP_H, CROP_W), 'float64')

  # Initialize the DCT feature extractor
  DCTF = bob.ip.DCTFeatures( BLOCK_H, BLOCK_W, OVERLAP_H, OVERLAP_W, N_DCT_COEF)

  # process the 'dictionary of files'
  for k in img_input:
    # input image file
    img_rgb = bob.io.load( str(img_input[k]) )
    # input eyes position file
    LW, LH, RW, RH = [int(j.strip()) for j in open(pos_input[k]).read().split()]

    # convert to grayscale
    img = bob.ip.rgb_to_gray(img_rgb)
    # extract and crop a face 
    FEN(img, cropped_img, LH, LW, RH, RW) 
    # preprocess a face using Tan and Triggs
    TT(cropped_img, preprocessed_img)
    # computes DCT features
    dct_blocks=dctfeatures(preprocessed_img, BLOCK_H, BLOCK_W, OVERLAP_H, OVERLAP_W, N_DCT_COEF,
      True, True, False)

    # save
    bob.io.Array(dct_blocks).save(str(features_output[k]))


def stats_computation(img_input, img_output, ubm):
  """Computes GMMStats against a world model"""
  
  ubm = bob.machine.GMMMachine(bob.io.HDF5File(ubm))
  gmmstats = bob.machine.GMMStats(ubm.dim_c, ubm.dim_d)

  # process the 'dictionary of files'
  for k in img_input:
    # input image file
    img = bob.io.Arrayset( str(img_input[k]) )
    # accumulates statistics
    gmmstats.init()
    ubm.acc_statistics(img, gmmstats)
    # save statistics
    gmmstats.save(bob.io.HDF5File( str(img_output[k]) ) ) 
  


def NormalizeStdArrayset(arrayset):
  arrayset.load()

  length = arrayset.shape[0]
  n_samples = len(arrayset)
  mean = numpy.ndarray(length, 'float64')
  std = numpy.ndarray(length, 'float64')

  mean.fill(0)
  std.fill(0)

  for array in arrayset:
    x = array.get().astype('float64')
    mean += x
    std += (x ** 2)

  mean /= n_samples
  std /= n_samples
  std -= (mean ** 2)
  std = std ** 0.5 # sqrt(std)

  arStd = bob.io.Arrayset()
  for array in arrayset:
    arStd.append(array.get().astype('float64') / std)

  return (arStd,std)


def multiplyVectorsByFactors(matrix, vector):
  for i in range(0, matrix.rows()):
    for j in range(0, matrix.columns()):
      matrix[i, j] *= vector[j]


def loadData(files):
  data = bob.io.Arrayset()
  for f in files:
    data.extend(bob.io.Array(str(f)).get())

  return data


def trainGMM(data, n_gaussians=32, iterk=25, iterg=25, convergence_threshold=1e-5, variance_threshold=0.001, 
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

  [variances, weights] = kmeans.get_variances_and_weights_for_each_cluster(normalizedAr)
  means = kmeans.means

  # Undo normalization
  if norm_KMeans:
    multiplyVectorsByFactors(means, stdAr)
    multiplyVectorsByFactors(variances, stdAr ** 2)

  # Initialize gmm
  gmm.means = means
  gmm.set_variance_thresholds(variance_threshold)
  gmm.variances = variances
  gmm.weights = weights

  # Train gmm
  trainer = bob.trainer.ML_GMMTrainer(update_means, update_variances, update_weights)
  trainer.convergenceThreshold = convergence_threshold
  trainer.maxIterations = iterg
  trainer.train(gmm, ar)

  return gmm



def adaptGMM(data, prior_gmm, iterg=25, convergence_threshold=1e-5, variance_threshold=0.001, adapt_weight=False, adapt_variance=False, relevance_factor=0.001, responsibilities_threshold=0, bob3_map=False, alpha_bob3=0.5):

  ar=data

  # Load prior gmm
  prior_gmm.set_variance_thresholds(variance_threshold)

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
  gmm.set_variance_thresholds(variance_threshold)

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
    self.relevance_factor = 4

  def precomputeZTnorm(self, tnorm_clients, znorm_clients):
    # Loading data for ZTnorm
    # Getting models for tnorm_clients

    i = 0
    self.tnorm_models = []
    for c in tnorm_clients:
      self.tnorm_models.append(self.getModel(c))
      i += 1

    # Getting statistics for znorm_clients

    tnorm_clients_ext=[]
    i = 0
    self.znorm_tests = []
    for c in znorm_clients:
      train_files = self.db.files(directory=self.features_dir, extension=self.extension, protocol=self.protocol, purposes='probe', model_ids=(c,), groups=None, classes='client')
      for f in train_files.itervalues():
        [file_basename, x] = os.path.splitext(os.path.basename(f))
        stat_path =  os.path.join(self.models_dir, "statswm_" + file_basename + "_" + str(c) + ".hdf5")
        if os.path.exists(stat_path):
          stats = bob.machine.GMMStats(bob.io.HDF5File(str(stat_path)))
        else:
          data = loadData([f])
          stats = bob.machine.GMMStats(self.wm.dim_c, self.wm.dim_d)
          stats.init()
          self.wm.acc_statistics(data, stats)
          stats.save(bob.io.HDF5File(str(stat_path)))

        self.znorm_tests.append(stats)
        #tnorm_clients_ext.append(c)
        r_id = self.db.get_real_id_from_tnorm_id(c)
        tnorm_clients_ext.append(r_id)

      i += 1


    self.D = bob.machine.linear_scoring(self.tnorm_models, self.wm, self.znorm_tests)
    tnorm_real_ids = []
    for c in tnorm_clients:
      r_id = self.db.get_real_id_from_tnorm_id(c)
      tnorm_real_ids.append(r_id)
    self.D_sameValue = self.sameValue(tnorm_real_ids, tnorm_clients_ext)

    # Loading data for ZTnorm ... done"
    
  def sameValue(self, vect_A, vect_B):
    sameMatrix = numpy.ndarray((len(vect_A), len(vect_B)), 'bool')

    for j in range(len(vect_A)):
      for i in range(len(vect_B)):
        sameMatrix[j, i] = (vect_A[j] == vect_B[i])

    return sameMatrix

  def setZTnormGroup(self, group):
      
    tnorm_clients = self.db.tnorm_ids(protocol=self.protocol)
    znorm_clients = self.db.tnorm_ids(protocol=self.protocol)

    self.precomputeZTnorm(tnorm_clients, znorm_clients)
    

  def train(self, model_id):
    # Training model
    train_files = self.db.files(directory=self.features_dir, extension=self.extension, protocol=self.protocol, purposes='enrol', model_ids=(model_id,), groups=None, classes=None)
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
        stats = bob.machine.GMMStats(self.wm.dim_c, self.wm.dim_d)
        self.wm.acc_statistics(data, stats)
        list_stats.append(stats)
      
      scores = bob.machine.linear_scoring(models, self.wm, list_stats)
    else:
      scores = numpy.ndarray((len(models), len(files)), 'float64')
      
      nb_scores = len(models)*len(files)
      i=0
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
          sys.stdout.flush()

    if self.ztnorm:
      # TODO: fix n_blocks
      n_blocks = 4161
      A = scores / n_blocks
      #print "A: " + str(A)
      B = bob.machine.linear_scoring(models, self.wm, self.znorm_tests) / n_blocks
      #print "B: " + str(B)
      C = bob.machine.linear_scoring(self.tnorm_models, self.wm, list_stats) / n_blocks 
      #print "C: " + str(C)
      scores = bob.machine.ztnorm(A, B, C, self.D/n_blocks, self.D_sameValue)
    return scores

  def convert_score_to_list(self, scores, probes):
    ret = []
    i = 0
    for c in probes.itervalues():
      ret.append((c[1], c[2], c[3], c[4], scores[0, i]))
      i+=1

    return ret

  def scores_client(self, model_id):
    client_probes = self.db.objects(directory=self.features_dir, extension=self.extension, protocol=self.protocol, purposes="probe", model_ids=(model_id,), classes="client") 

    files = [x[0] for x in client_probes.itervalues()]
    scores = self.scores([self.getModel(model_id)], files)
    
    return self.convert_score_to_list(scores, client_probes)

  def scores_impostor(self, model_id):
    client_probes = self.db.objects(directory=self.features_dir, extension=self.extension, protocol=self.protocol, purposes="probe", model_ids=(model_id,), classes="impostor")

    files = [x[0] for x in client_probes.itervalues()]
    scores = self.scores([self.getModel(model_id)], files)
    
    return self.convert_score_to_list(scores, client_probes)

  
  def run(self, groups, output_file):
    models = self.db.models(groups=groups)
    if self.ztnorm:
      self.setZTnormGroup(groups)

    i=0
    total=len(models)
    scores4 = numpy.ndarray(4, 'float64')
    for c in models:
      scores=self.scores_client(c)
      scores4[0] = scores[0][4]
      scores4[1] = scores[1][4]
      for x in scores:
        output_file.write(str(x[2]) + " " + str(x[0]) + " " + str(x[3]) + " " + str(x[4]) + "\n") 

      scores=self.scores_impostor(c)
      scores4[2] = scores[0][4]
      scores4[3] = scores[1][4]
      for x in scores:
        output_file.write(str(x[2]) + " " + str(x[0]) + " " + str(x[3]) + " " + str(x[4]) + "\n") 
      
      i+=1
    
    return scores4


def jfa_enrol(features_c, ubm, jfa_base, output_machine, n_iter):
  # Prepare list of GMMStats
  gmmstats_c = []
  # Initialize a python list for the GMMStats
  for k in features_c:
    # Process one file
    stats = bob.machine.GMMStats( bob.io.HDF5File(str(features_c[k])) )
    # append in the list
    gmmstats_c.append(stats)

  ubm_machine = bob.machine.GMMMachine(bob.io.HDF5File(ubm))
  base_machine = bob.machine.JFABaseMachine(bob.io.HDF5File(jfa_base))
  base_machine.ubm = ubm_machine
  machine = bob.machine.JFAMachine(base_machine)
  base_trainer = bob.trainer.JFABaseTrainer(base_machine)
  trainer = bob.trainer.JFATrainer(machine, base_trainer)
  trainer.enrol(gmmstats_c, n_iter)
  machine.save(bob.io.HDF5File(output_machine))

def compute_scores(db, group, gmmstats_dir, extension, protocol, clientmodel_dir, ubm, jfa_base, jfa_model_ext, jfa_enrol_n_iter):
  models = db.models(groups=group)
  client_scores = []
  impostor_scores = []
  ubm_machine = bob.machine.GMMMachine(bob.io.HDF5File(ubm))
  base_machine = bob.machine.JFABaseMachine(bob.io.HDF5File(jfa_base))
  base_machine.ubm = ubm_machine
  for c in models:
    # 1/ enrol
    features_c = db.files(directory=gmmstats_dir, extension=extension, protocol=protocol, model_ids=(c,), purposes='enrol')
    client_model_filename = os.path.join(clientmodel_dir,str(c)+str(jfa_model_ext))
    jfa_enrol(features_c, ubm, jfa_base, client_model_filename, jfa_enrol_n_iter)
    machine = bob.machine.JFAMachine(bob.io.HDF5File(client_model_filename))
    machine.jfa_base = base_machine

    # 2/ probe: client accesses
    client_probes = db.objects(directory=gmmstats_dir, extension=extension, protocol=protocol, purposes="probe", model_ids=(c,), classes="client")
    # compute the score for each probe
    scores_c = []
    for k in client_probes:
      probe = bob.machine.GMMStats( bob.io.HDF5File(str(client_probes[k][0])) )
      new_score = machine.forward(probe)
      scores_c.append( new_score )
    client_scores.extend(scores_c)

    # 3/ probe: impostor accesses
    client_probes = db.objects(directory=gmmstats_dir, extension=extension, protocol=protocol, purposes="probe", model_ids=(c,), classes="impostor")
    # compute the score for each probe
    scores_i = []
    for k in client_probes:
      probe = bob.machine.GMMStats( bob.io.HDF5File(str(client_probes[k][0])) )
      new_score = machine.forward(probe)
      scores_i.append( new_score )
    impostor_scores.extend(scores_i)

  return (client_scores, impostor_scores)


class TestBancaSmall(unittest.TestCase):
  """Performs various face recognition tests using the BANCA_SMALL database."""
  
  def test01_features(self):
    """Creates the features in a temporary directory"""
    # Creates a temporary directory
    output_dir = os.environ['BOB_FACE_VERIF_TEMP_DIRECTORY']
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    self.assertTrue( not os.path.exists(output_dir) )
    os.makedirs(output_dir)
    
    # define some database-related variables 
    db = bob.db.banca_small.Database()
    extension='.hdf5'

    # Computes the features
    img_input = db.files(directory='bancasmall', extension=".jpg")
    pos_input = db.files(directory='bancasmall', extension=".pos")
    features_dir = os.path.join(output_dir, 'features')
    if not os.path.exists(features_dir):
      os.mkdir(features_dir)
    features_output = db.files(directory=features_dir, extension=extension)
    face_normalized(img_input, pos_input, features_output)

  def test02_gmm_ztnorm(self):
    """GMM toolchain experiments with ZT-norm"""

    # define some database-related variables 
    db = bob.db.banca_small.Database()
    protocol='P'
    extension='.hdf5'
    output_dir = os.environ['BOB_FACE_VERIF_TEMP_DIRECTORY']
    features_dir = os.path.join(output_dir, 'features')

    # create a subdirectory for the models
    models_dir = os.path.join(output_dir, "models")
    if not os.path.exists(models_dir):
      os.mkdir(models_dir)

    # loads the UBM model
    wm_path = os.path.join('bancasmall', "ubmT5_new.hdf5")
    wm = bob.machine.GMMMachine(bob.io.HDF5File(wm_path))

    # creates a GMM experiments using Linear Scoring and ZT-norm
    exp = GMMExperiment(db, features_dir, extension, protocol, wm, models_dir, True, True)
    exp.iterg = 25
    exp.iterk = 25
    exp.convergence_threshold = 0.0005
    exp.variance_threshold = 0.001
    exp.relevance_factor = 4.
    exp.responsibilities_threshold = 0

    # creates a directory for the results
    result_dir=os.path.join(output_dir, "results")
    if not os.path.exists(result_dir):
      os.makedirs(result_dir)

    # Run the experiment
    scores=exp.run('dev', open(os.path.join(result_dir, 'scores-dev'), 'w'))

    # Check results (scores)
    #scores_ref = numpy.array([2.073368737400600, 1.524833680242284, 
    #  2.468051383113884, 1.705402816531652])
    scores_ref = numpy.array([0.91875557, 0.53932973, 1.48734588, 1.22941611])
    self.assertTrue( (abs(scores - scores_ref) < 1e-4).all() )

  def test03_jfa(self):
    """Tests JFA"""

    # define some database-related variables 
    db = bob.db.banca_small.Database()
    protocol='P'
    extension='.hdf5'
    output_dir = os.environ['BOB_FACE_VERIF_TEMP_DIRECTORY']
    features_dir = os.path.join(output_dir, 'features')
    gmmstats_dir = os.path.join(output_dir, 'gmmstats')
    if not os.path.exists(gmmstats_dir):
      os.makedirs(gmmstats_dir)

    # loads the UBM model
    wm_path = os.path.join('bancasmall', "ubmT5_new.hdf5")
    ubm_model = bob.machine.GMMMachine(bob.io.HDF5File(wm_path))

    img_input = db.files(directory=features_dir, extension=extension)
    img_output = db.files(directory=gmmstats_dir, extension=extension)
    stats_computation(img_input, img_output, wm_path)

    # JFA training
    jfa_ru = 3
    jfa_rv = 2 
    jfa_train_n_iter = 10
    jfa_train_relevance_factor = 4
    base_machine = bob.machine.JFABaseMachine(ubm_model, jfa_ru, jfa_rv) 
    T = bob.trainer.JFABaseTrainer(base_machine)
    Vinit = bob.io.load(os.path.join('bancasmall', 'jfa_Vinit.hdf5'))
    Uinit = bob.io.load(os.path.join('bancasmall', 'jfa_Uinit.hdf5'))
    Dinit = bob.io.load(os.path.join('bancasmall', 'jfa_Dinit.hdf5'))
    base_machine.v = Vinit
    base_machine.u = Uinit
    base_machine.d = Dinit
    # Load Training Data
    world_clients = db.clients(protocol=protocol, groups="world")
    gmmstats = []
    for c in world_clients:
      world_features_c = db.files(directory=gmmstats_dir, extension=extension, protocol=protocol, groups="world", real_ids=(c,))
      # Initialize a python list for the GMMStats
      gmmstats_c = []
      for k in world_features_c:
        # Process one file
        stats = bob.machine.GMMStats( bob.io.HDF5File(str(world_features_c[k])) )
        # append in the list
        gmmstats_c.append(stats)
      gmmstats.append(gmmstats_c)
    T.trainNoInit(gmmstats, jfa_train_n_iter)
    output_machine = os.path.join(output_dir, 'jfa_model_UVD.hdf5')
    base_machine.save(bob.io.HDF5File(output_machine))
    # TODO: compare trained and enrolled values?
 
    jfa_enrol_n_iter = 1 
    clientmodel_dir = os.path.join(output_dir, 'jfamodels')
    if not os.path.exists(clientmodel_dir):
      os.makedirs(clientmodel_dir)
    (client_scores, impostor_scores) = compute_scores(db, 'dev', gmmstats_dir, extension, protocol, clientmodel_dir, wm_path, output_machine, extension, jfa_enrol_n_iter)
    client_scores_ref = [0.0049256627092841053, 0.004753096187349203]
    impostor_scores_ref = [0.013874685706151616, 0.0065295654116636487]
    self.assertTrue(abs(client_scores[0] - client_scores_ref[0]) < 1e-7)    
    self.assertTrue(abs(client_scores[1] - client_scores_ref[1]) < 1e-7)    
    self.assertTrue(abs(impostor_scores[0] - impostor_scores_ref[0]) < 1e-7)
    self.assertTrue(abs(impostor_scores[1] - impostor_scores_ref[1]) < 1e-7) 


  def test04_cleanup(self):
    """Cleanup temporary directory"""

    # Remove output directory
    output_dir = os.environ['BOB_FACE_VERIF_TEMP_DIRECTORY']
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    self.assertTrue( not os.path.exists(output_dir) )

# Instantiates our standard main module for unittests
main = bob.helper.unittest_main(TestBancaSmall)
