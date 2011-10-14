/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Tue 11 oct 2011
 *
 * @brief Machines that implements the Probabilistic Linear Discriminant 
 *   Analysis Model of Prince and Helder,
 *   'Probabilistic Linear Discriminant Analysis for Inference About Identity',
 *   ICCV'2007
 */

#ifndef TORCH_MACHINE_PLDAMACHINE_H
#define TORCH_MACHINE_PLDAMACHINE_H

#include "io/HDF5File.h"

namespace Torch { namespace machine {
  
  /**
   * A PLDA Base machine which contains F, G and Sigma matrices as well as mu.
   */
  class PLDABaseMachine {

    public:

      /**
       * Default constructor. Builds an otherwise invalid 0x0 PLDA base 
       * machine.
       */
      PLDABaseMachine();

      /**
       * Constructor, builds a new PLDA machine. F, G, Sigma and mu are not 
       * initialized.
       *
       * @param d Dimensionality of the feature vector
       * @param nf size of F (d x nf)
       * @param ng size of G (d x ng)
       */ 
      PLDABaseMachine(const size_t d, const size_t nf, const size_t ng);

      /**
       * Copies another machine
       */
      PLDABaseMachine(const PLDABaseMachine& other);

      /**
       * Starts a new PLDABaseMachine from an existing Configuration object.
       */
      PLDABaseMachine(Torch::io::HDF5File& config);

      /**
       * Just to virtualize the destructor
       */
      virtual ~PLDABaseMachine(); 

      /**
       * Assigns from a different machine
       */
      PLDABaseMachine& operator= (const PLDABaseMachine &other);

      /**
       * Loads data from an existing configuration object. Resets the current
       * state.
       */
      void load(Torch::io::HDF5File& config);

      /**
       * Saves an existing machine to a Configuration object.
       */
      void save(Torch::io::HDF5File& config) const;

      /** 
       * Resizes the PLDA Machine. F, G, sigma and mu will should be 
       * considered uninitialized.
       */
      void resize(const size_t d, const size_t nf, const size_t ng);

      /**
        * Gets the F matrix
        */
      inline const blitz::Array<double,2>& getF() const 
      { return m_F; }

      /**
        * Sets the F matrix
        */
      void setF(const blitz::Array<double,2>& F);

      /**
       * Returns the current F matrix in order to be updated.
       * @warning Use with care. Only trainers should use this function for
       * efficiency reasons.
       */
      inline blitz::Array<double, 2>& updateF()  
      { return m_F; }

      /**
        * Gets the G matrix
        */
      inline const blitz::Array<double,2>& getG() const 
      { return m_G; }

      /**
        * Sets the G matrix
        */
      void setG(const blitz::Array<double,2>& G);

      /**
       * Returns the current G matrix in order to be updated.
       * @warning Use with care. Only trainers should use this function for
       * efficiency reasons.
       */
      inline blitz::Array<double, 2>& updateG()
      { return m_G; }

      /**
        * Gets the Sigma (diagonal) 'matrix'
        */
      inline const blitz::Array<double,1>& getSigma() const 
      { return m_sigma; }

      /**
        * Sets the Sigma matrix
        */
      void setSigma(const blitz::Array<double,1>& s);

      /**
       * Returns the current Sigma matrix in order to be updated.
       * @warning Use with care. Only trainers should use this function for
       * efficiency reasons.
       */
      inline blitz::Array<double, 1>& updateSigma()
      { return m_sigma; }

      /**
        * Gets the mu vector
        */
      inline const blitz::Array<double,1>& getMu() const 
      { return m_mu; }

      /**
        * Sets the mu vector
        */
      void setMu(const blitz::Array<double,1>& mu);

      /**
       * Returns the current mu vector in order to be updated.
       * @warning Use with care. Only trainers should use this function for
       * efficiency reasons.
       */
      inline blitz::Array<double, 1>& updateMu()
      { return m_mu; }


      /**
        * Gets the feature dimensionality
        */
      inline size_t getDimD() const 
      { return m_mu.extent(0); }

      /**
        * Gets the size/rank the F matrix along the second dimension
        */
      inline size_t getDimF() const 
      { return m_F.extent(1); }

      /**
        * Gets the size/rank the G matrix along the second dimension
        */
      inline size_t getDimG() const 
      { return m_G.extent(1); }


    private:
      // F, G and Sigma matrices, and mu vector
      // Sigma is assumed to be diagonal, and only the diagonal is stored
      blitz::Array<double,2> m_F;
      blitz::Array<double,2> m_G;
      blitz::Array<double,1> m_sigma; 
      blitz::Array<double,1> m_mu; 
  };


}}

#endif
