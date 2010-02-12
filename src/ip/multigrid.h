#ifndef _TORCH5SPRO_MULTIGRID_H_
#define _TORCH5SPRO_MULTIGRID_H_

#include "Tensor.h"

// used for the coefficients
#define CENTER 0
#define LEFT 1
#define RIGHT 2
#define UP 3
#define DOWN 4

#define IS_NEAR(var, value, delta) ((var >= (value - delta)) && (var <= (value + delta)))

namespace Torch {

/** @name Useful functions for the multigrid framework
 
    @author Guillaume Heusch (heusch@idiap.ch)
    @author Laurent El Shafey (laurent.el-shafey@idiap.ch)
    @version 2.0
    \Date
    @since 2.0
*/
//@{

  /** intergrid transfer function: fine-to-coarse projection (full weighting)
    
  @param data a tensor vector defined on a fine grid 
  @param restricted a tensor vector on the coarse grid (result)
  */
  void restriction(const DoubleTensor& data, DoubleTensor& restricted );
  
  /** intergrid transfer function: coarse-to-fine projection (bilinear interpolation)
      
  @param data a tensor vector defined on a coarse grid 
  @param restricted a tensor vector on the fine grid (result)
  */   
  void project(const DoubleTensor& data, DoubleTensor& projected );
  
  /** build the differential operator (i.e. the sparse matrix induced by the discretization of the PDE

  @param matrix  the differential operator (result)
  @param rho the diffusion coefficients 
  @param lambda the relative importance of the smoothness constraint
  @param type the the kind of diffusion performed (isotropic, anisotropic)
  @param image the data
  */
  void buildOperator(DoubleTensor& matrix, DoubleTensor& rho, const double lambda, const int type, const DoubleTensor& image);
 
  /** compute the diffusion coefficients for each point of the data
      
  @param rho the diffusion coefficients 
  @param image the image 
  @param x the x-coordinate of the processed point
  @param y the y-coordinate of the processed point
  @param type the the kind of diffusion performed (isotropic, anisotropic)
  */
  void computeCoeff(DoubleTensor& rho, const DoubleTensor& image, const int x, const int y, const int type);
  
  /** perform Gauss-Seidel relaxation on Ax=b

  @param source is the right hand side term of the above equation (b)
  @param rho the diffusion coefficients 
  @param lambda the relative importance of the smoothness constraint
  @param type the the kind of diffusion performed (isotropic, anisotropic)
  */
  void gaussSeidel(DoubleTensor& result, const DoubleTensor& source, DoubleTensor& rho, const double lambda, const int type );

  /** perform Jacobi relaxation on Ax=b

  @param source is the right hand side term of the above equation (b)
  @param rho the diffusion coefficients 
  @param lambda the relative importance of the smoothness constraint
  @param type the the kind of diffusion performed (isotropic, anisotropic)
  */
  void jacobi(DoubleTensor& result, const DoubleTensor& source, DoubleTensor& rho, const double lambda, const int type); 
  
  /// custom matrix multiplication (takes advantage of the sparse structure of the differential operator)
  void myMultiply(const DoubleTensor& data, DoubleTensor& result, DoubleTensor& rho, const double lambda, const int type);

  /** compute, for each point, the diffusion coefficients (weber contrast)
  
  @param rho array with the coefficients for a point (result)
  @param image the data
  @param x the x-coordinate of the processed point
  @param y the y-coordinate of the processed point
  @param position direction in which the coefficient is computed (up, down, left, right)
  */
  void weber(DoubleTensor& rho, const DoubleTensor& image, const int x, const int y, const char *position); 

  //@}
}
#endif

