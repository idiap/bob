#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Laurent El Shafey <Laurent.El-Shafey@idiap.ch>

"""Test trainer package
"""
import os, sys, tempfile, shutil, math
import unittest
import torch

def normalizeBlocks(src):
  for i in range(src.extent(0)):
    block = src[i, :, :]
    mean = torch.core.array.float64_2.mean(block)
    std = torch.core.array.float64_2.sum((block - mean) ** 2) / block.size()
    if std == 0:
      std = 1
    else:
      std = math.sqrt(std)

    src[i, :, :] = (block - mean) / std
    
def normalizeDCT(src):
  for i in range(src.extent(1)):
    col = src[:, i]
    mean = torch.core.array.float64_1.mean(col)
    std = torch.core.array.float64_1.sum((col - mean) ** 2) / col.size()
    if std == 0:
      std = 1
    else:
      std = math.sqrt(std)

    src[:, i] = (col - mean) / std


def dctfeatures(prep, A_BLOCK_H, A_BLOCK_W, A_OVERLAP_H, A_OVERLAP_W, 
    A_N_DCT_COEF, norm_before, norm_after, add_xy):
  
  blockShape = torch.ip.getBlockShape(prep, A_BLOCK_H, A_BLOCK_W, A_OVERLAP_H, A_OVERLAP_W)
  blocks = torch.core.array.float64_3(blockShape)
  torch.ip.block(prep, blocks, A_BLOCK_H, A_BLOCK_W, A_OVERLAP_H, A_OVERLAP_W)

  if norm_before:
    normalizeBlocks(blocks)

  if add_xy:
    real_DCT_coef = A_N_DCT_COEF - 2
  else:
    real_DCT_coef = A_N_DCT_COEF

  
  # Initialize cropper and destination array
  DCTF = torch.ip.DCTFeatures(A_BLOCK_H, A_BLOCK_W, A_OVERLAP_H, A_OVERLAP_W, real_DCT_coef)
  
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
  
  TMP_tensor = torch.core.array.float64_2(n_blocks, TMP_tensor_max)
  
  nBlocks = torch.ip.getNBlocks(prep, A_BLOCK_H, A_BLOCK_W, A_OVERLAP_H, A_OVERLAP_W)
  for by in range(nBlocks[0]):
    for bx in range(nBlocks[1]):
      bi = bx + by * nBlocks[1]
      if add_xy:
        TMP_tensor[bi, 0] = bx
        TMP_tensor[bi, 1] = by
      
      TMP_tensor[bi, TMP_tensor_min:TMP_tensor_max] = dct_blocks[bi, dct_blocks_min:dct_blocks_max]

  if norm_after:
    normalizeDCT(TMP_tensor)

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
  FEN = torch.ip.FaceEyesNorm( CROP_EYES_D, CROP_H, CROP_W, CROP_OH, CROP_OW)
  cropped_img = torch.core.array.float64_2(CROP_H, CROP_W)

  # Initialize the Tan and Triggs preprocessing
  TT = torch.ip.TanTriggs( GAMMA, SIGMA0, SIGMA1, SIZE, THRESHOLD, ALPHA)
  preprocessed_img = torch.core.array.float64_2(CROP_H, CROP_W)

  # Initialize the DCT feature extractor
  DCTF = torch.ip.DCTFeatures( BLOCK_H, BLOCK_W, OVERLAP_H, OVERLAP_W, N_DCT_COEF)

  # process the 'dictionary of files'
  for k in img_input:
    # input image file
    img_rgb = torch.core.array.load( str(img_input[k]) )
    # input eyes position file
    LW, LH, RW, RH = [int(j.strip()) for j in open(pos_input[k]).read().split()]

    # convert to grayscale
    img = torch.ip.rgb_to_gray(img_rgb)
    # extract and crop a face 
    FEN(img, cropped_img, LH, LW, RH, RW) 
    # preprocess a face using Tan and Triggs
    TT(cropped_img, preprocessed_img)
    # computes DCT features
    dct_blocks=dctfeatures(preprocessed_img, BLOCK_H, BLOCK_W, OVERLAP_H, OVERLAP_W, N_DCT_COEF,
      True, True, False)

    # save
    torch.io.Array(dct_blocks).save(str(features_output[k]))


def NormalizeStdArrayset(arrayset):
  arrayset.load()

  length = arrayset.shape[0]
  n_samples = len(arrayset)
  mean = torch.core.array.float64_1(length)
  std = torch.core.array.float64_1(length)

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

  arStd = torch.io.Arrayset()
  for array in arrayset:
    arStd.append(array.get().cast('float64') / std)

  return (arStd,std)


def multiplyVectorsByFactors(matrix, vector):
  for i in range(0, matrix.rows()):
    for j in range(0, matrix.columns()):
      matrix[i, j] *= vector[j]


def loadData(files):
  data = torch.io.Arrayset()
  for f in files:
    data.extend(torch.io.Array(str(f)))

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
  kmeans = torch.machine.KMeansMachine(n_gaussians, input_size)
  gmm = torch.machine.GMMMachine(n_gaussians, input_size)

  # Create the KMeansTrainer
  kmeansTrainer = torch.trainer.KMeansTrainer()
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
  gmm.setVarianceThresholds(variance_threshold)
  gmm.variances = variances
  gmm.weights = weights

  # Train gmm
  trainer = torch.trainer.ML_GMMTrainer(update_means, update_variances, update_weights)
  trainer.convergenceThreshold = convergence_threshold
  trainer.maxIterations = iterg
  trainer.train(gmm, ar)

  return gmm



def adaptGMM(data, prior_gmm, iterg=25, convergence_threshold=1e-5, variance_threshold=0.001, adapt_weight=False, adapt_variance=False, relevance_factor=0.001, responsibilities_threshold=0, torch3_map=False, alpha_torch3=0.5):

  ar=data

  # Load prior gmm
  prior_gmm.setVarianceThresholds(variance_threshold)

  # Create trainer
  if responsibilities_threshold == 0.:
    trainer = torch.trainer.MAP_GMMTrainer(relevance_factor, True, adapt_variance, adapt_weight)
  else:
    trainer = torch.trainer.MAP_GMMTrainer(relevance_factor, True, adapt_variance, adapt_weight, responsibilities_threshold)
  trainer.convergenceThreshold = convergence_threshold
  trainer.maxIterations = iterg
  trainer.setPriorGMM(prior_gmm)

  if torch3_map:
    trainer.setT3MAP(alpha_torch3)

  # Load gmm
  gmm = torch.machine.GMMMachine(prior_gmm)
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
          stats = torch.machine.GMMStats(torch.io.HDF5File(str(stat_path)))
        else:
          data = loadData([f])
          stats = torch.machine.GMMStats(self.wm.nGaussians, self.wm.nInputs)
          stats.init()
          self.wm.accStatistics(data, stats)
          stats.save(torch.io.HDF5File(str(stat_path)))

        self.znorm_tests.append(stats)
        #tnorm_clients_ext.append(c)
        r_id = self.db.getRealIdFromTNormId(c)
        tnorm_clients_ext.append(r_id)

      i += 1


    self.D = torch.machine.linearScoring(self.tnorm_models, self.wm, self.znorm_tests)
    tnorm_real_ids = []
    for c in tnorm_clients:
      r_id = self.db.getRealIdFromTNormId(c)
      tnorm_real_ids.append(r_id)
    self.D_sameValue = self.sameValue(tnorm_real_ids, tnorm_clients_ext)

    # Loading data for ZTnorm ... done"
    
  def sameValue(self, vect_A, vect_B):
    sameMatrix = torch.core.array.bool_2(len(vect_A), len(vect_B))

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
        self.client_models[model_id] = torch.machine.GMMMachine(torch.io.HDF5File(model_path))
      else:
        self.client_models[model_id] = self.train(model_id)
        self.client_models[model_id].save(torch.io.HDF5File(model_path))
    
    return self.client_models[model_id]
    

  def scores(self, models, files):
    if self.linear_scoring:
      list_stats=[]
      for f in files :
        data = loadData([f])
        stats = torch.machine.GMMStats(self.wm.nGaussians, self.wm.nInputs)
        self.wm.accStatistics(data, stats)
        list_stats.append(stats)
      
      scores = torch.machine.linearScoring(models, self.wm, list_stats)
    else:
      scores = torch.core.array.float64_2(len(models), len(files))
      
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
      B = torch.machine.linearScoring(models, self.wm, self.znorm_tests) / n_blocks
      C = torch.machine.linearScoring(self.tnorm_models, self.wm, list_stats) / n_blocks 
      scores = torch.machine.ztnorm(A, B, C, self.D/n_blocks, self.D_sameValue)
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
    scores4 = torch.core.array.float64_1((4,))
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


class TestTest(unittest.TestCase):
  """Performs various face recognition tests using the BANCA_SMALL database."""
  
  def test01_gmm_ztnorm(self):
    # Creates a temporary directory
    output_dir = tempfile.mkdtemp()

    # Get the directory where the features and the UBM are stored
    data_dir = os.path.join('data', 'bancasmall')
    
    # define some database-related variables 
    db = torch.db.banca_small.Database()
    protocol='P'
    extension='.hdf5'

    # Computes the features
    img_input = db.files(directory=data_dir, extension=".jpg")
    pos_input = db.files(directory=data_dir, extension=".pos")
    features_dir = os.path.join(output_dir, 'features')
    if not os.path.exists(features_dir):
      os.mkdir(features_dir)
    features_output = db.files(directory=features_dir, extension=extension)
    face_normalized(img_input, pos_input, features_output)

    # create a subdirectory for the models
    models_dir = os.path.join(output_dir, "models")
    if not os.path.exists(models_dir):
      os.mkdir(models_dir)

    # loads the UBM model
    wm_path = os.path.join(data_dir, "ubmT5_new.hdf5")
    wm = torch.machine.GMMMachine(torch.io.HDF5File(wm_path))

    # creates a GMM experiments using Linear Scoring and ZT-norm
    exp = GMMExperiment(db, features_dir, extension, protocol, wm, models_dir, True, True)
    exp.iterg = 25
    exp.iterk = 25
    exp.convergence_threshold = 0.0005
    exp.variance_threshold = 0.001
    exp.relevance_factor = 4.
    exp.responsibilities_threshold = 0

    # creates a directory for the results
    result_dir=os.path.join(output_dir, "results", "")
    if not os.path.exists(result_dir):
      os.makedirs(result_dir)

    # Run the experiment
    scores=exp.run('dev', open(os.path.join(result_dir, 'scores-dev'), 'w'))

    # Check results (scores)
    #scores_ref = torch.core.array.float64_1([2.073368737400600, 1.524833680242284, 
    #  2.468051383113884, 1.705402816531652], (4,))
    scores_ref = torch.core.array.float64_1([1.46379478, 0.69330295, 2.14465708, 1.24284387], (4,))
    self.assertTrue( ((scores - scores_ref) < 1e-4).all() )

    # Remove output directory
    shutil.rmtree(output_dir)


if __name__ == '__main__':
  sys.argv.append('-v')
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStart'):
    torch.core.ProfilerStart(os.environ['TORCH_PROFILE'])
  os.chdir(os.path.realpath(os.path.dirname(sys.argv[0])))
  unittest.main()
  if os.environ.has_key('TORCH_PROFILE') and \
      os.environ['TORCH_PROFILE'] and \
      hasattr(torch.core, 'ProfilerStop'):
    torch.core.ProfilerStop()
