/**
 * @author Laurent El-Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Thu 14 Jul 2011 16:08:30
 *
 * @brief JFA functions
 */

#ifndef TORCH5SPRO_TRAINER_JFATRAINER_H
#define TORCH5SPRO_TRAINER_JFATRAINER_H

#include <blitz/array.h>
#include <map>
#include <vector>
#include <string>
#include "core/array_check.h"
#include "io/Arrayset.h"
#include "machine/JFAMachine.h"

#include "core/logging.h"

namespace Torch { namespace trainer { namespace jfa {

/**
  C++ port of the JFA Cookbook
*/

/**
  @brief Updates eigenchannels (or eigenvoirces) from accumulators A and C. 

  @param A  An array of 2D square matrices (c x ru x ru)
  @param C  A 2D matrix (ru x cd)
  @param uv A 2D matrix (ru x cd)

  @warning  ru: rank of the matrix of eigenchannels u
            cd: size of the supervector
*/
void updateEigen(const blitz::Array<double,3> &A, const blitz::Array<double,2> &C,
  blitz::Array<double,2> &uv);

/**
  @brief Provides new estimates of channel factors x given zeroth and first 
    order sufficient statistics (N. F), current hyper-parameters of joint 
    factor analysis  model (m, E, d, u, v) and current estimates of speaker 
    and channel factors (x, y, z), for a joint factor analysis model.

  @param F 2D matrix (T x CD) of first order statistics (not centered). 
    The rows correspond to training segments (T segments). 
    The number of columns is given by the supervector dimensionality. 
    The first n columns correspond to the n dimensions of the first Gaussian 
    component, the second n columns to second component, and so on.

  @param N 2D matrix (TxC) of zero order statistics (occupation counts of 
    Gaussian components). 
    The rows correspond to training segments. 
    The columns correspond to Gaussian components.

  @param m 1D speaker and channel independent mean supervector (CD)
    (e.g. concatenated UBM mean vectors)

  @param E 1D speaker and channel independent variance supervector (CD)
    (e.g. concatenated UBM variance vectors)

  @param d 1D row vector (CD) that is the diagonal from the diagonal matrix 
    describing the remaining speaker variability (not described by 
    eigenvoices). 
    The number of columns is given by the supervector dimensionality.

  @param v 2D matrix of eigenvoices (rv x CD)
    The rows of matrix v are 'eigenvoices'. (The number of rows must be the
    same as the number of columns of matrix y). 
    The number of columns is given by the supervector dimensionality.

  @param u 2D matrix of eigenchannels (ru x CD)
    The rows of matrix u are 'eigenchannels'. (The number of rows must be the
    same as the number of columns of matrix x).
    Number of columns is given by the supervector dimensionality.

   @param z 2D matrix of speaker factors corresponding to matrix d (Nspeaker x CD)
    The rows correspond to speakers (values in vector spk_ids are the indices 
    of the rows, therfore the number of the rows must be (at least) the highest
    value in spk_ids). Number of columns is given by the supervector 
    dimensionality.
 
  @param y 2D matrix of speaker factors corresponding to eigenvoices (Nspeaker x rv)
    The rows correspond to speakers (values in vector spk_ids are the indices 
    of the rows, therfore the number of the rows must be (at least) the 
    highest value in spk_ids). 
    The columns correspond to eigenvoices (The number of columns must the same 
    as the number of rows of matrix v).

  @param x 2D matrix of speaker factors for eigenchannels (Nspeaker x ru)
    The rows correspond to training segments. 
    The columns correspond to eigenchannels (The number of columns must be 
    the same as the number of rows of matrix u)

  @param spk_ids 1D column vector (T) with rows corresponding to training 
    segments and integer values identifying a speaker. Rows having same values
    identifies segments spoken by same speakers. 
    The values are indices of rows in y and z matrices containing 
    corresponding speaker factors.
  @warning Rows corresponding to the same speaker SHOULD be consecutive.
*/
void estimateXandU(const blitz::Array<double,2> &F, const blitz::Array<double,2> &N,
  const blitz::Array<double,1> &m, const blitz::Array<double,1> &E, 
  const blitz::Array<double,1> &d, const blitz::Array<double,2> &v, 
  const blitz::Array<double,2> &u, const blitz::Array<double,2> &z, 
  const blitz::Array<double,2> &y, blitz::Array<double,2> &x,
  const blitz::Array<uint32_t,1> &spk_ids);



void estimateYandV(const blitz::Array<double,2> &F, const blitz::Array<double,2> &N,
  const blitz::Array<double,1> &m, const blitz::Array<double,1> &E, 
  const blitz::Array<double,1> &d, const blitz::Array<double,2> &v, 
  const blitz::Array<double,2> &u, const blitz::Array<double,2> &z, 
  blitz::Array<double,2> &y, const blitz::Array<double,2> &x,
  const blitz::Array<uint32_t,1> &spk_ids);

void estimateZandD(const blitz::Array<double,2> &F, const blitz::Array<double,2> &N,
  const blitz::Array<double,1> &m, const blitz::Array<double,1> &E, 
  const blitz::Array<double,1> &d, const blitz::Array<double,2> &v, 
  const blitz::Array<double,2> &u, blitz::Array<double,2> &z, 
  const blitz::Array<double,2> &y, const blitz::Array<double,2> &x,
  const blitz::Array<uint32_t,1> &spk_ids);
} // close JFA namespace



class JFABaseTrainer {

  public:
    /**
     * Initializes a new JFABaseTrainer. This implementation is consistent with the 
     * JFA cookbook implementation. 
     */
    JFABaseTrainer(Torch::machine::JFABaseMachine& jfa_machine);

    /**
     * Destructor virtualisation
     */
    ~JFABaseTrainer() {}

    /**
     * Set the zeroth and first order statistics to train
     * @warning the JFABaseMachine should be set before
     */
    void setStatistics(const std::vector<blitz::Array<double,2> >& N, 
      const std::vector<blitz::Array<double,2> >& F);

    /**
     * Set the x, y, z speaker factors
     * @warning the JFABaseMachine should be set before
     */
    void setSpeakerFactors(const std::vector<blitz::Array<double,2> >& x, 
      const std::vector<blitz::Array<double,1> >& y,
      const std::vector<blitz::Array<double,1> >& z);

    /**
     * Initializes randomly the U, V and D=diag(d) matrices of the JFA machine
     */
    void initializeRandomU();
    void initializeRandomV();
    void initializeRandomD();


    /**** Y and V functions ****/
    /**
     * Computes Vt * diag(sigma)^-1
     */
    void computeVtSigmaInv();
    /**
     * Computes Vt_{c} * diag(sigma)^-1 * V_{c} for each Gaussian c
     */
    void computeVProd();
    /**
     * Computes (I+Vt*diag(sigma)^-1*Ni*V)^-1 which occurs in the y estimation
     * for the given person
     */
    void computeIdPlusVProd_i(const int id);
    /**
     * Computes sum_{sessions h}(N_{i,h}*(o_{i,h} - m - D*z_{i} - U*x_{i,h}) 
     * which occurs in the y estimation of the given person
     */
    void computeFn_y_i(const int id);
    /**
     * Updates y_i (of the current person) and the accumulators to compute V 
     * with the cache values m_cache_IdPlusVprod_i, m_VtSigmaInv and 
     * m_cache_Fn_y_i
     */
    void updateY_i(const int id);
    /**
     * Updates y and the accumulators to compute V 
     */
    void updateY();
    /**
     * Updates V by using the accumulators m_cache_A1_x and m_cache_A2_x
     * V = A2 * A1^-1
     */
    void updateV();


    /**** X and U functions ****/
    /**
     * Computes Ut * diag(sigma)^-1
     */
    void computeUtSigmaInv();
    /**
     * Computes Ut_{c} * diag(sigma)^-1 * U_{c} for each Gaussian c
     */
    void computeUProd();
    /**
     * Computes (I+Vt*diag(sigma)^-1*Ni*V)^-1 which occurs in the y estimation
     * for the given person
     */
    void computeIdPlusUProd_ih(const int id, const int h);
    /**
     * Computes sum_{sessions h}(N_{i,h}*(o_{i,h} - m - D*z_{i} - U*x_{i,h}) 
     * which occurs in the y estimation of the given person
     */
    void computeFn_x_ih(const int id, const int h);
    /**
     * Updates x_ih (of the current person/session) and the accumulators to compute V 
     * with the cache values m_cache_IdPlusVprod_i, m_VtSigmaInv and 
     * m_cache_Fn_y_i
     */
    void updateX_ih(const int id, const int h);
    /**
     * Updates x and the accumulators to compute U
     */
    void updateX();
    /**
     * Updates U by using the accumulators m_cache_A1_y and m_cache_A2_y
     * U = A2 * A1^-1
     */
    void updateU();


    /**** z and D functions ****/
    /**
     * Computes diag(D) * diag(sigma)^-1
     */
    void computeDtSigmaInv();
    /**
     * Computes Dt_{c} * diag(sigma)^-1 * D_{c} for each Gaussian c
     */
    void computeDProd();
    /**
     * Computes (I+diag(d)t*diag(sigma)^-1*Ni*diag(d))^-1 which occurs in the z estimation
     * for the given person
     */
    void computeIdPlusDProd_i(const int id);
    /**
     * Computes sum_{sessions h}(N_{i,h}*(o_{i,h} - m - V*y_{i} - U*x_{i,h}) 
     * which occurs in the y estimation of the given person
     */
    void computeFn_z_i(const int id);
    /**
     * Updates z_i (of the current person) and the accumulators to compute D
     * with the cache values m_cache_IdPlusDProd_i, m_VtSigmaInv and 
     * m_cache_Fn_z_i
     */
    void updateZ_i(const int id);
    /**
     * Updates z and the accumulators to compute D
     */
    void updateZ();
    /**
     * Updates D by using the accumulators m_cache_A1_z and m_cache_A2_z
     * V = A2 * A1^-1
     */
    void updateD();



    /**
     * Precomputes the sums of the zeroth order statistics over the sessions 
     * for each client
     */
    void precomputeSumStatisticsN();
    /**
     * Precomputes the sums of the first order statistics over the sessions 
     * for each client
     */
    void precomputeSumStatisticsF();



    /**
      * Trains the Joint Factor Analysis by initializing U, V, and D randomly
      */
    void train(const std::vector<blitz::Array<double,2> >& N,
      const std::vector<blitz::Array<double,2> >& F, 
      const size_t n_iter); 
    void train(const std::vector<std::vector<const Torch::machine::GMMStats*> >& features,
      const size_t n_iter); 
    /**
      * Trains the oint Factor Analysis without initializing U, V and D
      */
    void trainNoInit(const std::vector<blitz::Array<double,2> >& N,
      const std::vector<blitz::Array<double,2> >& F, 
      const size_t n_iter); 
    void trainNoInit(const std::vector<std::vector<const Torch::machine::GMMStats*> >& features,
      const size_t n_iter); 

    /**
      * Trains the Inter Session Variability model by initializing U randomly
      */
    void trainISV(const std::vector<blitz::Array<double,2> >& N,
      const std::vector<blitz::Array<double,2> >& F, 
      const size_t n_iter, const double relevance_factor); 
    void trainISV(const std::vector<std::vector<const Torch::machine::GMMStats*> >& features,
      const size_t n_iter, const double relevance_factor); 
    /**
      * Trains the Inter Session Variability model without initializing U
      */
    void trainISVNoInit(const std::vector<blitz::Array<double,2> >& N,
      const std::vector<blitz::Array<double,2> >& F, 
      const size_t n_iter, const double relevance_factor); 
    void trainISVNoInit(const std::vector<std::vector<const Torch::machine::GMMStats*> >& features,
      const size_t n_iter, const double relevance_factor); 

    /**
      * Initializes V=zero and D=srqt(var_UBM / rel_factor) for ISV
      */
    void initializeVD_ISV(const double relevance_factor);
    /**
      * Initializes U, V and D
      */
    void initializeUVD();
    /**
      * Initializes X, Y and Z
      */
    void initializeXYZ();

    /**
     * Get the zeroth order statistics
     */
    const std::vector<blitz::Array<double,2> >& getN() const
    { return m_N; }
    const std::vector<blitz::Array<double,1> >& getNacc() const
    { return m_Nacc; }
    /**
     * Get the first order statistics
     */
    const std::vector<blitz::Array<double,2> >& getF() const
    { return m_F; }
    const std::vector<blitz::Array<double,1> >& getFacc() const
    { return m_Facc; }
    /**
     * Get the x speaker factors
     */
    const std::vector<blitz::Array<double,2> >& getX() const
    { return m_x; }
    /**
     * Get the y speaker factors
     */
    const std::vector<blitz::Array<double,1> >& getY() const
    { return m_y; }
    /**
     * Get the z speaker factors
     */
    const std::vector<blitz::Array<double,1> >& getZ() const
    { return m_z; }

    /**
      * Set the zeroth order statistics
      */
    void setN(const std::vector<blitz::Array<double,2> >& N)
    { m_N = N; }
    /**
      * Set the first order statistics
      */
    void setF(const std::vector<blitz::Array<double,2> >& F)
    { m_F = F; }
    /**
      * Set the x speaker factors
      */
    void setX(const std::vector<blitz::Array<double,2> >& X)
    { m_x = X; }
    /**
      * Set the y speaker factors
      */
    void setY(const std::vector<blitz::Array<double,1> >& y)
    { m_y = y; }
    /**
      * Set the z speaker factors
      */
    void setZ(const std::vector<blitz::Array<double,1> >& z)
    { m_z = z; }


    /**
      * For debugging only
      */
    const blitz::Array<double,2>& getVtSigmaInv() const
    { return m_cache_VtSigmaInv; }
    const blitz::Array<double,2>& getIdPlusVProd_i() const
    { return m_cache_IdPlusVProd_i; }
    const blitz::Array<double,1>& getFn_y_i() const
    { return m_cache_Fn_y_i; }
    const blitz::Array<double,3>& getA1_y() const
    { return m_cache_A1_y; }
    const blitz::Array<double,2>& getA2_y() const
    { return m_cache_A2_y; }
    blitz::Array<double,2>& updateVtSigmaInv()
    { return m_cache_VtSigmaInv; }
    blitz::Array<double,2>& updateIdPlusVProd_i()
    { return m_cache_IdPlusVProd_i; }
    blitz::Array<double,1>& updateFn_y_i()
    { return m_cache_Fn_y_i; }
    blitz::Array<double,3>& updateA1_y()
    { return m_cache_A1_y; }
    blitz::Array<double,2>& updateA2_y()
    { return m_cache_A2_y; }
    void setVtSigmaInv(const blitz::Array<double,2>& VtSigmaInv)
    { m_cache_VtSigmaInv.reference(Torch::core::array::ccopy(VtSigmaInv)); }
    void setIdPlusVProd_i(const blitz::Array<double,2>& IdPlusVProd_i)
    { m_cache_IdPlusVProd_i.reference(Torch::core::array::ccopy(IdPlusVProd_i)); }
    void setFn_y_i(const blitz::Array<double,1>& Fn_y_i)
    { m_cache_Fn_y_i.reference(Torch::core::array::ccopy(Fn_y_i)); }
    void setA1_y(const blitz::Array<double,3>& A1_y)
    { m_cache_A1_y.reference(Torch::core::array::ccopy(A1_y)); }
    void setA2_y(const blitz::Array<double,2>& A2_y)
    { m_cache_A2_y.reference(Torch::core::array::ccopy(A2_y)); }


  private:
    /**
     * Initializes randomly a 1D vector
     */
    void initializeRandom(blitz::Array<double,1>& vector);
    /**
     * Initializes randomly a 2D matrix
     */
    void initializeRandom(blitz::Array<double,2>& matrix);


    Torch::machine::JFABaseMachine& m_jfa_machine; // JFABaseMachine
    size_t m_Nid; // Number of identities
    std::vector<blitz::Array<double,2> > m_N; // Zeroth order statistics
    std::vector<blitz::Array<double,2> > m_F; // First order statistics

    std::vector<blitz::Array<double,2> > m_x; // matrix x of speaker factors for eigenchannels U, for each client
    std::vector<blitz::Array<double,1> > m_y; // vector y of spealer factors for eigenvoices V, for each client
    std::vector<blitz::Array<double,1> > m_z; // vector z of spealer factors for eigenvoices Z, for each client

    // Cache/Precomputation
    std::vector<blitz::Array<double,1> > m_Nacc; // Sum of the zeroth order statistics over the sessions for each client
    std::vector<blitz::Array<double,1> > m_Facc; // Sum of the first order statistics over the sessions for each client

    blitz::Array<double,1> m_cache_ubm_mean;
    blitz::Array<double,1> m_cache_ubm_var;

    blitz::Array<double,2> m_cache_VtSigmaInv; // Vt * diag(sigma)^-1
    blitz::Array<double,3> m_cache_VProd; // first dimension is the Gaussian id
    blitz::Array<double,2> m_cache_IdPlusVProd_i;
    blitz::Array<double,1> m_cache_Fn_y_i;
    blitz::Array<double,3> m_cache_A1_y;
    blitz::Array<double,2> m_cache_A2_y;

    blitz::Array<double,2> m_cache_UtSigmaInv; // Ut * diag(sigma)^-1
    blitz::Array<double,3> m_cache_UProd; // first dimension is the Gaussian id
    blitz::Array<double,2> m_cache_IdPlusUProd_ih;
    blitz::Array<double,1> m_cache_Fn_x_ih;
    blitz::Array<double,3> m_cache_A1_x;
    blitz::Array<double,2> m_cache_A2_x;

    blitz::Array<double,1> m_cache_DtSigmaInv; // Dt * diag(sigma)^-1
    blitz::Array<double,1> m_cache_DProd; // supervector length dimension
    blitz::Array<double,1> m_cache_IdPlusDProd_i;
    blitz::Array<double,1> m_cache_Fn_z_i;
    blitz::Array<double,1> m_cache_A1_z;
    blitz::Array<double,1> m_cache_A2_z;

    mutable blitz::Array<double,2> m_tmp_rvrv;
    mutable blitz::Array<double,2> m_tmp_rvCD;
    mutable blitz::Array<double,2> m_tmp_rvD;
    mutable blitz::Array<double,2> m_tmp_ruD;
    mutable blitz::Array<double,2> m_tmp_ruru;
    mutable blitz::Array<double,1> m_tmp_rv;
    mutable blitz::Array<double,1> m_tmp_ru;
    mutable blitz::Array<double,1> m_tmp_CD;
    mutable blitz::Array<double,1> m_tmp_CD_b;
    mutable blitz::Array<double,1> m_tmp_D;
    mutable blitz::Array<double,2> m_tmp_CDCD;
};


class JFATrainer {

  public:
    /**
     * Initializes a new JFA trainer. This implementation is consistent with the 
     * JFA cookbook implementation. 
     */
    JFATrainer(Torch::machine::JFAMachine& jfa_machine, Torch::trainer::JFABaseTrainer& base_trainer);

    /**
     * Destructor virtualisation
     */
    ~JFATrainer() {}

    /**
      * Main procedures for enroling with Joint Factor Analysis
      */
    void enrol(const blitz::Array<double,2>& N, const blitz::Array<double,2>& F,
      const size_t n_iter);
    void enrol(const std::vector<const Torch::machine::GMMStats*>& features,
      const size_t n_iter);

  private:

    Torch::machine::JFAMachine& m_jfa_machine; // JFAMachine
    Torch::trainer::JFABaseTrainer& m_base_trainer; // JFABaseTrainer
};


}}

#endif /* TORCH5SPRO_TRAINER_JFATRAINER_H */
