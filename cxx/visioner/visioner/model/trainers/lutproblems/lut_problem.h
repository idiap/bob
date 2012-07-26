#ifndef BOB_VISIONER_LUT_PROBLEM_H
#define BOB_VISIONER_LUT_PROBLEM_H

#include "visioner/model/dataset.h"
#include "visioner/model/model.h"
#include "visioner/model/loss.h"

namespace bob { namespace visioner {

  class LUTProblem;
  typedef boost::shared_ptr<LUTProblem>  rlutproblem_t;
  typedef std::vector<rlutproblem_t>     rlutproblems_t;

  ////////////////////////////////////////////////////////////////////////////////
  // LUTProblem: 
  //      greedily train LUTs in boosting rounds.
  ////////////////////////////////////////////////////////////////////////////////

  class LUTProblem
  {
    public:

      // Constructor
      LUTProblem(const DataSet& dataset, const param_t& param);

      // Destructor
      virtual ~LUTProblem() {}

      // Update predictions
      void update_scores(const LUTs& luts);

      // Update loss values and derivatives
      virtual void update_loss_deriv() = 0;
      virtual void update_loss() = 0;

      // Select the optimal feature
      virtual void select() = 0;

      // Optimize the LUT entries for the selected feature
      bool line_search();

      // Compute the loss value/error
      virtual scalar_t value() const = 0;
      virtual scalar_t error() const = 0;

      // Compute the gradient <g> and the function value in the <x> point
      //      (used during linesearch)
      virtual scalar_t linesearch(const scalar_t* x, scalar_t* g) = 0;

      // Access functions
      index_t n_entries() const { return m_data.n_fvalues(); }
      index_t n_features() const { return m_data.n_features(); }
      index_t n_samples() const { return m_data.n_samples(); }
      index_t n_outputs() const { return m_data.n_outputs(); }

      discrete_t fvalue(index_t f, index_t s) const { return m_data.value(f, s); }
      const scalar_t* target(index_t s) const { return m_data.targets()[s]; }
      scalar_t cost(index_t s) const { return m_data.cost(s); }

      const MultiLUTs& mluts() const { return m_mluts; }
      const LUTs& luts() const { return m_luts; }

    protected:

      // Update current scores
      void update_cscores(const scalar_t* x);

    protected:

      // Attributes
      const DataSet&          m_data;         // Dataset
      param_t                 m_param;

      const rloss_t           m_rloss;        // Base loss
      const Loss&             m_loss;         

      FeatureSharingType      m_sharing;      // Feature sharing method

      MultiLUTs		m_mluts;	// Trained model
      LUTs			m_luts;		// Buffered LUTs

      scalar_mat_t            m_sscores;	// Strong learner's score: (sample, output)
      scalar_mat_t		m_wscores;	// Weak learner's score: (sample, output)
      scalar_mat_t            m_cscores;      // Current (strong + scale * weak) scores: (sample, output)

      scalar_mat_t            m_umasks;       // Entries mask [0/1]: (feature, entry)
  };

}}

#endif // BOB_VISIONER_LUT_PROBLEM_H
