from ..core import __from_extension_import__
__from_extension_import__('._ip', __package__, locals(), ['GLCMProp'])

def properties_by_name(self, glcm_matrix, prop_names=None):
  """Possibility to query the properties of GLCM by specifying a name. Returns a list of numpy.array of the queried properties.
  
    glcm The input GLCM as 3D numpy.ndarray of dtype='float64'
    prop_names A list GLCM texture properties' names
  """
  prop_dict = {"angular second moment":self.angular_second_moment, 
               "energy":self.energy, 
               "variance":self.variance,
               "contrast":self.contrast,
               "autocorrelation":self.auto_correlation,
               "correlation":self.correlation,
               "correlation matlab":self.correlation_m,
               "inverse difference moment":self.inv_diff_mom,
               "sum average":self.sum_avg,
               "sum variance":self.sum_var,
               "sum entropy":self.sum_entropy,
               "entropy":self.entropy,
               "difference variance":self.diff_var,
               "difference entropy":self.diff_entropy,
               "dissimilarity":self.dissimilarity,
               "homogeneity":self.homogeneity,
               "cluster prominance":self.cluster_prom,
               "cluster shade":self.cluster_shade,
               "maximum probability":self.max_prob,
               "information measure of correlation 1":self.inf_meas_corr1,
               "information measure of correlation 2":self.inf_meas_corr2,               
               "inverse difference":self.inv_diff,
               "inverse difference normalized":self.inv_diff_norm,
               "inverse difference moment normalized":self.inv_diff_mom_norm
               }
  if prop_names == None:
    prop_names = prop_dict.keys() 
  retval = []
  for props in prop_names:
    retval.append(prop_dict[props](glcm_matrix))

  return retval    
  
GLCMProp.properties_by_name = properties_by_name
del properties_by_name      
