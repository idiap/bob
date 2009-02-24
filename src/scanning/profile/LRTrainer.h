#ifndef _TORCHVISION_SCANNING_PROFILE_LR_TRAINER_H_
#define _TORCHVISION_SCANNING_PROFILE_LR_TRAINER_H_

namespace Torch
{
namespace Profile
{
//	class Classifier;
//
//	/////////////////////////////////////////////////////////////////////////
//	// Torch::ipSWEvaluator:
//	//	- use some 2-class classifier to check if some sub-window contains a pattern
//	//      - the data passed to the classifier is buffered in a tensor of the same size as the model
//	//              and of the same type as the input tensor
//	//
//	//      - MULTISCALE approach (the input size is larger than the model's):
//        //              - rescale the input tensor to the buffer, presuming
//        //                      the input tensor has an integral image like format
//        //      - PYRAMID approach (the input size is the same as the model's):
//        //              - crop the input tensor to the buffer (actually just copy the values)
//        //
//        //      - PARAMETERS (name, type, default value, description):
//        //              "saveBuffTensorToJpg"     bool    false   "save the buffer tensor to JPEG");
//	//
//	// TODO: doxygen header!
//	/////////////////////////////////////////////////////////////////////////
//
//	class ipSWEvaluator : public ipSubWindow
//	{
//	public:
//
//		// Constructor
//		ipSWEvaluator();
//
//		// Destructor
//		virtual ~ipSWEvaluator();
//
//		// Set the classifier to load from some file
//		bool                    setClassifier(const char* filename);
//
//		/// Change the sub-window to process in - overriden
//		/// Checks also if there is some pattern in this sub-window
//		virtual bool		setSubWindow(int sw_x, int sw_y, int sw_w, int sw_h);
//
//		/////////////////////////////////////////////////////////////////
//		// Access functions
//
//		// Get the result - the sub-window contains the pattern?!
//		bool			isPattern() const;
//		//	... get the model confidence of this
//		double			getConfidence() const;
//
//		// Get the model size
//		int		        getModelWidth() const;
//		int	        	getModelHeight() const;
//
//		/////////////////////////////////////////////////////////////////
//
//	protected:
//
//                /////////////////////////////////////////////////////////////////
//
//                /// Check if the input tensor has the right dimensions and type - overriden
//		virtual bool		checkInput(const Tensor& input) const;
//
//		/// Allocate (if needed) the output tensors given the input tensor dimensions - overriden
//		virtual bool		allocateOutput(const Tensor& input);
//
//		/// Process some input tensor (the input is checked, the outputs are allocated) - overriden
//		/// The classifier should be loaded first!
//		virtual bool		processInput(const Tensor& input);
//
//		/// called when some option was changed - overriden
//		virtual void		optionChanged(const char* name);
//
//		/////////////////////////////////////////////////////////////////
//
//        private:
//
//                /////////////////////////////////////////////////////////////////
//
//                // Crop some input tensor to the model size (it's actually just copying)
//                void                    cropInput(const Tensor& input);
//
//                // Scale some input tensor (considered integral image) to the model size
//                void                    iscaleInput(const Tensor& input);
//
//		/////////////////////////////////////////////////////////////////
//		// Attributes
//
//		// Machine used for deciding if some sub-window contains a pattern or not
//		Classifier*             m_classifier;
//
//		// Buffer tensor to pass to the classifier
//		Tensor*                 m_buffTensor;
//		int*                    m_buff_indexes;         // Precomputed indexes in the buffer tensor
//		int                     m_buff_n_indexes;
//
//		// User for fast computing the offset of the scanning window
//		int                     m_input_stride_w;
//		int                     m_input_stride_h;
//		int                     m_input_stride_p;
//
//		// Crop/Copy case - precomputed index for the source (image/features to scan)
//		//      correlated with the indexes for the buffer tensor
//		int*                    m_copy_indexes;         // Precomputed indexes in the input tensor
//
//		// Scale case - precomputed indexes
//		// (the pixel values in the buffer corresponds to the
//		//      ((bottomright + topleft) - (topright + bottomlef)) / cellsize
//		//      in the integral image input tensor)
//		int*                    m_scale_br_indexes;     // botom right
//		int*                    m_scale_tl_indexes;     // top left
//		int*                    m_scale_tr_indexes;     // top right
//		int*                    m_scale_bl_indexes;     // bottom left
//		int*                    m_scale_cell_sizes;     // cell size
//
//		// Save buffered tensor to jpeg
//		bool                    m_save_buffTensor;
//
//		// Keep a copy of the input tensor (to pass to setSubWindow)
//                const Tensor*           m_input_copy;
//	};
}
}

#endif
