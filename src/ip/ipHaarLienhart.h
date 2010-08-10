#ifndef _TORCH5SPRO_IP_HAAR_LIENHART_H_
#define _TORCH5SPRO_IP_HAAR_LIENHART_H_

#include "ip/ipCore.h"
#include "core/Tensor.h"
#include "sp/spCores.h"

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::ipHaar_Lienhart
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////
	// There can be two conditions
	// One is take the offset from ipSubwindow, and take the width and height from the feature extraction.
	// Other option is to take the  window width and height and resize to model width and height and then extract feature

	class ipHaarLienhart : public ipCore
	{
	public:

		// Constructor
		ipHaarLienhart();//int width_, int height_);
		bool setNoRec(int noRecs_); //set the number of rectangles
		bool setRec(int Rec_,double weight_, int y_, int x_, int h_, int w_);

		/// Change the region of the input tensor to process - overriden
		virtual void		setRegion(const TensorRegion& region);

		/// Change the model size (if used with some machine) - overriden
		virtual void		setModelSize(const TensorSize& modelSize);

		// Get the ID specific to each spCore - overriden
		virtual int		getID() const { return IP_HAAR_LIENHART_ID; }

		/// Constructs an empty spCore of this kind - overriden
		/// (used by \c spCoreManager, this object is automatically deallocated)
		virtual spCore*		getAnInstance() const { return manage(new ipHaarLienhart()); }

		// Destructor
		virtual ~ipHaarLienhart();

		/////////////////////////////////////////////////////////////////
		virtual bool 		saveFile(File& file) const;
		virtual bool 		loadFile(File& file);
		protected:

		/////////////////////////////////////////////////////////////////

		/// Check if the input tensor has the right dimensions and type - overriden
		virtual bool		checkInput(const Tensor& input) const;

		/// Allocate (if needed) the output tensors given the input tensor dimensions - overriden
		virtual bool		allocateOutput(const Tensor& input);

		/// Process some input tensor (the input is checked, the outputs are allocated) - overriden
		virtual bool		processInput(const Tensor& input);

		/////////////////////////////////////////////////////////////////

	protected:

		void updateParameters();
		/////////////////////////////////////////////////////////////////
		// Attributes
		int m_noRecs; //Number of Rectangles


		int m_width;
		int m_height;

		int m_nparams;
		int *m_parameters;
		// Format for parameters

		double *m_weight;
		//    DoubleTensor *t_;
		//  DoubleTensor *t__;

		int u_x,u_y, u_size_x,u_size_y;
		int u_size_z,u_z; //for selecting the dimension
		int *u_parameters;
		double *u_weight;
		double invscale;
	};

	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	// REGISTER this spCore to the \c spCoreManager
	const bool ip_haar_lienhart_registered = spCoreManager::getInstance().add(
		manage(new ipHaarLienhart()), "Lienhart Haar");
	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
}

#endif



