#include "FLDATrainer.h"


//#include "Classifier.h"
//#include "Machines.h"
//#include "Tensor.h"
//#include "Image.h"
//#include "xtprobeImageFile.h"
//
///////////////////////////////////////////////////////////////////////////////////////////////////
//// Function to allocate a 2D/3D buffer tensor of the model size
////      and compute the matching indexes with the input tensor
///////////////////////////////////////////////////////////////////////////////////////////////////
//
//#define SW_EVAL_ALLOC(tensorType)                                                       \
//{                                                                                       \
//        if (input.nDimension() == 3)                                                    \
//        {                                                                               \
//                SW_EVAL_ALLOC_3D(tensorType);                                           \
//        }                                                                               \
//        else                                                                            \
//        {                                                                               \
//                SW_EVAL_ALLOC_2D(tensorType);                                           \
//        }                                                                               \
//}
//
///////////////////////////////////////////////////////////////////////////////////////////////////
//// Function to allocate a 3D buffer tensor of the model size
////      and compute the matching indexes with the input tensor
///////////////////////////////////////////////////////////////////////////////////////////////////
//
//#define SW_EVAL_ALLOC_3D(tensorType)                                                    \
//{                                                                                       \
//        const int n_planes = input.size(2);                                             \
//                                                                                        \
//        const tensorType* t_src = (tensorType*)&input;                                  \
//                                                                                        \
//        const int src_stride_h = t_src->t->stride[0];	                                \
//	const int src_stride_w = t_src->t->stride[1];	                                \
//	const int src_stride_p = t_src->t->stride[2];	                                \
//	m_input_stride_h = src_stride_h;                                                \
//	m_input_stride_w = src_stride_w;                                                \
//	m_input_stride_p = src_stride_p;                                                \
//                                                                                        \
//        tensorType* t_dst = new tensorType(model_h, model_w, n_planes);                 \
//        m_buffTensor = t_dst;                                                           \
//                                                                                        \
//        const int dst_stride_h = t_dst->t->stride[0];                                   \
//        const int dst_stride_w = t_dst->t->stride[1];                                   \
//        const int dst_stride_p = t_dst->t->stride[2];                                   \
//                                                                                        \
//        m_buff_n_indexes = model_h * model_w * n_planes;                                \
//        m_copy_indexes = new int[m_buff_n_indexes];                                     \
//        m_buff_indexes = new int[m_buff_n_indexes];                                     \
//                                                                                        \
//        int index = 0;                                                                  \
//        for (int y = 0; y < model_h; y ++)                                              \
//                for (int x = 0; x < model_w; x ++)                                      \
//                        for (int p = 0; p < n_planes; p ++)                             \
//                        {                                                               \
//                                m_copy_indexes[index] =                                 \
//                                        y * src_stride_h +                              \
//                                        x * src_stride_w +                              \
//                                        p * src_stride_p;                               \
//                                m_buff_indexes[index] =                                 \
//                                        y * dst_stride_h +                              \
//                                        x * dst_stride_w +                              \
//                                        p * dst_stride_p;                               \
//                                                                                        \
//                                index ++;                                               \
//                        }                                                               \
//}
//
///////////////////////////////////////////////////////////////////////////////////////////////////
//// Function to allocate a 2D buffer tensor of the model size
////      and compute the matching indexes with the input tensor
///////////////////////////////////////////////////////////////////////////////////////////////////
//
//#define SW_EVAL_ALLOC_2D(tensorType)                                                    \
//{                                                                                       \
//        const tensorType* t_src = (tensorType*)&input;                                  \
//                                                                                        \
//        const int src_stride_h = t_src->t->stride[0];	                                \
//	const int src_stride_w = t_src->t->stride[1];	                                \
//	m_input_stride_h = src_stride_h;                                                \
//	m_input_stride_w = src_stride_w;                                                \
//                                                                                        \
//        tensorType* t_dst = new tensorType(model_h, model_w);                           \
//        m_buffTensor = t_dst;                                                           \
//                                                                                        \
//        const int dst_stride_h = t_dst->t->stride[0];                                   \
//        const int dst_stride_w = t_dst->t->stride[1];                                   \
//                                                                                        \
//        m_buff_n_indexes = model_h * model_w;                                           \
//        m_copy_indexes = new int[m_buff_n_indexes];                                     \
//        m_buff_indexes = new int[m_buff_n_indexes];                                     \
//                                                                                        \
//        int index = 0;                                                                  \
//        for (int y = 0; y < model_h; y ++)                                              \
//                for (int x = 0; x < model_w; x ++)                                      \
//                {                                                                       \
//                        m_copy_indexes[index] =                                         \
//                                y * src_stride_h +                                      \
//                                x * src_stride_w;                                       \
//                        m_buff_indexes[index] =                                         \
//                                y * dst_stride_h +                                      \
//                                x * dst_stride_w;                                       \
//                                                                                        \
//                        index ++;                                                       \
//                }                                                                       \
//}
//
///////////////////////////////////////////////////////////////////////////////////////////////////
//// Function to copy a 2D/3D tensor to the buffer tensor of the model size
////      using the precomputed indexes
///////////////////////////////////////////////////////////////////////////////////////////////////
//
//#define SW_EVAL_COPY(tensorType, dataType)                                              \
//{                                                                                       \
//        const tensorType* t_src = (tensorType*)&input;                                  \
//        const dataType* src = t_src->t->storage->data + t_src->t->storageOffset;        \
//                                                                                        \
//        tensorType* t_dst = (tensorType*)m_buffTensor;                                  \
//        dataType* dst = t_dst->t->storage->data + t_dst->t->storageOffset;              \
//                                                                                        \
//        const int offset = m_sw_y * m_input_stride_h + m_sw_x * m_input_stride_w;       \
//                                                                                        \
//        for (int i = 0; i < m_buff_n_indexes; i ++)                                     \
//        {                                                                               \
//                dst[m_buff_indexes[i]] = src[m_copy_indexes[i] + offset];               \
//        }                                                                               \
//}
//
///////////////////////////////////////////////////////////////////////////////////////////////////
//// Function to compute the scalling coefficients for a 2D/3D buffer tensor
///////////////////////////////////////////////////////////////////////////////////////////////////
//
//#define SW_EVAL_PRESCALE(tensorType)                                                    \
//{                                                                                       \
//        if (m_buffTensor->nDimension() == 3)                                            \
//        {                                                                               \
//                SW_EVAL_PRESCALE_3D(tensorType);                                        \
//        }                                                                               \
//        else                                                                            \
//        {                                                                               \
//                SW_EVAL_PRESCALE_2D(tensorType);                                        \
//        }                                                                               \
//}
//
///////////////////////////////////////////////////////////////////////////////////////////////////
//// Function to compute the scalling coefficients for a 3D buffer tensor
///////////////////////////////////////////////////////////////////////////////////////////////////
//
//#define SW_EVAL_PRESCALE_3D(tensorType)                                                 \
//{                                                                                       \
//        const int n_planes = m_buffTensor->size(2);                                     \
//        const tensorType* t_dst = (tensorType*)m_buffTensor;                            \
//                                                                                        \
//        m_scale_br_indexes = new int[m_buff_n_indexes];                                 \
//        m_scale_tl_indexes = new int[m_buff_n_indexes];                                 \
//        m_scale_tr_indexes = new int[m_buff_n_indexes];                                 \
//        m_scale_bl_indexes = new int[m_buff_n_indexes];                                 \
//        m_scale_cell_sizes = new int[m_buff_n_indexes];                                 \
//                                                                                        \
//        const int model_w = getModelWidth();                              		\
//        const int model_h = getModelHeight();                             		\
//        const double inv_model_w = 1.0 / (model_w + 0.0);                               \
//        const double inv_model_h = 1.0 / (model_h + 0.0);                               \
//                                                                                        \
//        const double scale_w = 0.5 * (sw_w + 0.0) / (model_w + 0.0);                    \
//        const double scale_h = 0.5 * (sw_h + 0.0) / (model_h + 0.0);                    \
//                                                                                        \
//        int index = 0;                                                                  \
//        for (int y = 0; y < model_h; y ++)                                              \
//                for (int x = 0; x < model_w; x ++)                                      \
//                {                                                                       \
//                        const double x_in_sw = (x + 0.5) * inv_model_w * sw_w;          \
//                        const double y_in_sw = (y + 0.5) * inv_model_h * sw_h;          \
//                                                                                        \
//                        const double min_x_in_sw = x_in_sw - scale_w - 0.5;             \
//                        const double max_x_in_sw = x_in_sw + scale_w + 0.5;             \
//                        const double min_y_in_sw = y_in_sw - scale_h - 0.5;             \
//                        const double max_y_in_sw = y_in_sw + scale_h + 0.5;             \
//                                                                                        \
//                        const int l = getInRange((int)(min_x_in_sw), 0, sw_w - 1);      \
//                        const int r = getInRange((int)(max_x_in_sw + 0.5), 0, sw_w - 1);\
//                        const int t = getInRange((int)(min_y_in_sw), 0, sw_h - 1);      \
//                        const int b = getInRange((int)(max_y_in_sw + 0.5), 0, sw_h - 1);\
//                                                                                        \
//                        const int cell_size = (b - t) * (r - l);                        \
//                                                                                        \
//                        for (int p = 0; p < n_planes; p ++)                             \
//                        {                                                               \
//                                m_scale_br_indexes[index] =                             \
//                                        b * m_input_stride_h +                          \
//                                        r * m_input_stride_w +                          \
//                                        p * m_input_stride_p;                           \
//                                m_scale_tl_indexes[index] =                             \
//                                        t * m_input_stride_h +                          \
//                                        l * m_input_stride_w +                          \
//                                        p * m_input_stride_p;                           \
//                                m_scale_tr_indexes[index] =                             \
//                                        t * m_input_stride_h +                          \
//                                        r * m_input_stride_w +                          \
//                                        p * m_input_stride_p;                           \
//                                m_scale_bl_indexes[index] =                             \
//                                        b * m_input_stride_h +                          \
//                                        l * m_input_stride_w +                          \
//                                        p * m_input_stride_p;                           \
//                                m_scale_cell_sizes[index] = cell_size;                  \
//                                                                                        \
//                                index ++;                                               \
//                        }                                                               \
//                }                                                                       \
//}
//
///////////////////////////////////////////////////////////////////////////////////////////////////
//// Function to compute the scalling coefficients for a 2D buffer tensor
///////////////////////////////////////////////////////////////////////////////////////////////////
//
//#define SW_EVAL_PRESCALE_2D(tensorType)                                                 \
//{                                                                                       \
//        const tensorType* t_dst = (tensorType*)m_buffTensor;                            \
//                                                                                        \
//        m_scale_br_indexes = new int[m_buff_n_indexes];                                 \
//        m_scale_tl_indexes = new int[m_buff_n_indexes];                                 \
//        m_scale_tr_indexes = new int[m_buff_n_indexes];                                 \
//        m_scale_bl_indexes = new int[m_buff_n_indexes];                                 \
//        m_scale_cell_sizes = new int[m_buff_n_indexes];                                 \
//                                                                                        \
//        const int model_w = getModelWidth();                              		\
//        const int model_h = getModelHeight();                             		\
//        const double inv_model_w = 1.0 / (model_w + 0.0);                               \
//        const double inv_model_h = 1.0 / (model_h + 0.0);                               \
//                                                                                        \
//        const double scale_w = 0.5 * (sw_w + 0.0) / (model_w + 0.0);                    \
//        const double scale_h = 0.5 * (sw_h + 0.0) / (model_h + 0.0);                    \
//                                                                                        \
//        int index = 0;                                                                  \
//        for (int y = 0; y < model_h; y ++)                                              \
//                for (int x = 0; x < model_w; x ++)                                      \
//                {                                                                       \
//                        const double x_in_sw = (x + 0.5) * inv_model_w * sw_w;          \
//                        const double y_in_sw = (y + 0.5) * inv_model_h * sw_h;          \
//                                                                                        \
//                        const double min_x_in_sw = x_in_sw - scale_w - 0.5;             \
//                        const double max_x_in_sw = x_in_sw + scale_w + 0.5;             \
//                        const double min_y_in_sw = y_in_sw - scale_h - 0.5;             \
//                        const double max_y_in_sw = y_in_sw + scale_h + 0.5;             \
//                                                                                        \
//                        const int l = getInRange((int)(min_x_in_sw), 0, sw_w - 1);      \
//                        const int r = getInRange((int)(max_x_in_sw + 0.5), 0, sw_w - 1);\
//                        const int t = getInRange((int)(min_y_in_sw), 0, sw_h - 1);      \
//                        const int b = getInRange((int)(max_y_in_sw + 0.5), 0, sw_h - 1);\
//                                                                                        \
//                        const int cell_size = (b - t) * (r - l);                        \
//                                                                                        \
//                        m_scale_br_indexes[index] =                                     \
//                                b * m_input_stride_h +                                  \
//                                r * m_input_stride_w;                                   \
//                        m_scale_tl_indexes[index] =                                     \
//                                t * m_input_stride_h +                                  \
//                                l * m_input_stride_w;                                   \
//                        m_scale_tr_indexes[index] =                                     \
//                                t * m_input_stride_h +                                  \
//                                r * m_input_stride_w;                                   \
//                        m_scale_bl_indexes[index] =                                     \
//                                b * m_input_stride_h +                                  \
//                                l * m_input_stride_w;                                   \
//                        m_scale_cell_sizes[index] = cell_size;                          \
//                                                                                        \
//                        index ++;                                                       \
//                }                                                                       \
//}
//
///////////////////////////////////////////////////////////////////////////////////////////////////
//// Function to scale a 2D/3D tensor to the buffer tensor of the model size
////      using the precomputed indexes
///////////////////////////////////////////////////////////////////////////////////////////////////
//
//#define SW_EVAL_SCALE(tensorType, dataType)                                             \
//{                                                                                       \
//        const tensorType* t_src = (tensorType*)&input;                                  \
//        const dataType* src = (const dataType*)t_src->dataR();        			\
//                                                                                        \
//        tensorType* t_dst = (tensorType*)m_buffTensor;                                  \
//        dataType* dst = (dataType*)t_dst->dataW();					\
//                                                                                        \
//        const int offset = m_sw_y * m_input_stride_h + m_sw_x * m_input_stride_w;       \
//											\
//        for (int i = 0; i < m_buff_n_indexes; i ++)                                     \
//        {                                                                               \
//                dst[m_buff_indexes[i]] =                                                \
//                        (       src[m_scale_br_indexes[i] + offset] +                   \
//                                src[m_scale_tl_indexes[i] + offset] -                   \
//                                src[m_scale_tr_indexes[i] + offset] -                   \
//                                src[m_scale_bl_indexes[i] + offset])    /               \
//                        m_scale_cell_sizes[i];                                          \
//        }                                                                               \
//}
//
//namespace Torch
//{
//
///////////////////////////////////////////////////////////////////////////
//// Constructor
//
//ipSWEvaluator::ipSWEvaluator()
//	: 	ipSubWindow(),
//                m_classifier(0),
//
//                m_buffTensor(0),
//                m_buff_indexes(0), m_buff_n_indexes(0),
//
//                m_input_stride_w(0), m_input_stride_h(0), m_input_stride_p(0),
//
//                m_copy_indexes(0),
//
//                m_scale_br_indexes(0),
//                m_scale_tl_indexes(0),
//                m_scale_tr_indexes(0),
//                m_scale_bl_indexes(0),
//                m_scale_cell_sizes(0),
//
//                m_save_buffTensor(false),
//
//                m_input_copy(0)
//{
//        addBOption("saveBuffTensorToJpg", false, "save the buffer tensor to JPEG");
//}
//
///////////////////////////////////////////////////////////////////////////
//// Destructor
//
//ipSWEvaluator::~ipSWEvaluator()
//{
//        delete m_classifier;
//
//        delete m_buffTensor;
//        delete[] m_buff_indexes;
//
//        delete[] m_copy_indexes;
//
//        delete[] m_scale_br_indexes;
//        delete[] m_scale_tl_indexes;
//        delete[] m_scale_tr_indexes;
//        delete[] m_scale_bl_indexes;
//        delete[] m_scale_cell_sizes;
//}
//
///////////////////////////////////////////////////////////////////////////
//// called when some option was changed - overriden
//
//void ipSWEvaluator::optionChanged(const char* name)
//{
//        m_save_buffTensor = getBOption("saveBuffTensorToJpg");
//}
//
///////////////////////////////////////////////////////////////////////////
//// Set the classifier to load from some file
//
//bool ipSWEvaluator::setClassifier(const char* filename)
//{
//        // Load the machine
//        Machine* machine = Torch::loadMachineFromFile(filename);
//        if (machine == 0)
//        {
//                Torch::message("ipSWEvaluator::setClassifier - invalid model file!\n");
//                return false;
//        }
//
//        // Check if it's really a classifier
//        Classifier* classifier = dynamic_cast<Classifier*>(machine);
//        if (classifier == 0)
//        {
//                delete machine;
//                Torch::message("ipSWEvaluator::setClassifier - the loaded model is not a classifier!\n");
//                return false;
//        }
//
//        // OK
//        delete m_classifier;
//        m_classifier = classifier;
//        return true;
//}
//
///////////////////////////////////////////////////////////////////////////
//// Access functions
//
//inline bool ipSWEvaluator::isPattern() const
//{
//        if (m_classifier == 0)
//        {
//                Torch::error("ipSWEvaluator::isPattern - no valid classifier specified!\n");
//        }
//        return m_classifier->isPattern();
//}
//
//inline double ipSWEvaluator::getConfidence() const
//{
//        if (m_classifier == 0)
//        {
//                Torch::error("ipSWEvaluator::getConfidence - no valid classifier specified!\n");
//        }
//        return m_classifier->getConfidence();
//}
//
//inline int ipSWEvaluator::getModelWidth() const
//{
//        if (m_classifier == 0)
//        {
//                Torch::error("ipSWEvaluator::getModelWidth - no valid classifier specified!\n");
//        }
//        return m_classifier->getInputSize().size[1];
//}
//
//inline int ipSWEvaluator::getModelHeight() const
//{
//        if (m_classifier == 0)
//        {
//                Torch::error("ipSWEvaluator::getModelHeight - no valid classifier specified!\n");
//        }
//        return m_classifier->getInputSize().size[0];
//}
//
///////////////////////////////////////////////////////////////////////////
//// Check if the input tensor has the right dimensions and type - overriden
//
//bool ipSWEvaluator::checkInput(const Tensor& input) const
//{
//        return  m_classifier != 0 &&
//                (input.nDimension() == 2 || input.nDimension() == 3);
//}
//
///////////////////////////////////////////////////////////////////////////
//// Allocate (if needed) the output tensors given the input tensor dimensions - overriden
//
//bool ipSWEvaluator::allocateOutput(const Tensor& input)
//{
//        // No output is generated, the Machine has the output!
//        return true;
//}
//
///////////////////////////////////////////////////////////////////////////
//// Process some input tensor (the input is checked, the outputs are allocated) - overriden
//
//bool ipSWEvaluator::processInput(const Tensor& input)
//{
//        // Cleanup
//        delete m_buffTensor;
//        delete[] m_buff_indexes;
//        m_buffTensor = 0;
//        m_buff_indexes = 0;
//        m_buff_n_indexes = 0;
//
//        delete[] m_copy_indexes;
//        m_copy_indexes = 0;
//        m_input_stride_w = 0;
//        m_input_stride_h = 0;
//        m_input_stride_p = 0;
//
//        const int model_w = getModelWidth();
//        const int model_h = getModelHeight();
//
//        // Allocate the buffer tensor as to have the model size
//        //      and the input type and number of planes
//        //      and compute the indexes between the input tensor and the buffered one
//        switch (input.getDatatype())
//        {
//        case Tensor::Char:
//                SW_EVAL_ALLOC(CharTensor);
//                break;
//
//        case Tensor::Short:
//                SW_EVAL_ALLOC(ShortTensor);
//                break;
//
//        case Tensor::Int:
//		SW_EVAL_ALLOC(IntTensor);
//                break;
//
//        case Tensor::Long:
//                SW_EVAL_ALLOC(LongTensor);
//                break;
//
//        case Tensor::Float:
//                SW_EVAL_ALLOC(FloatTensor);
//                break;
//
//        case Tensor::Double:
//                SW_EVAL_ALLOC(DoubleTensor);
//                break;
//
//        default:
//                return false;
//        }
//
//        // Keep a copy of the input tensor
//        m_input_copy = &input;
//
//        // OK
//        return true;
//}
//
///////////////////////////////////////////////////////////////////////////
//// Crop some input tensor to the model size (it's actually just copying)
//
//void ipSWEvaluator::cropInput(const Tensor& input)
//{
//        switch (input.getDatatype())
//        {
//        case Tensor::Char:
//                SW_EVAL_COPY(CharTensor, char);
//                break;
//
//        case Tensor::Short:
//                SW_EVAL_COPY(ShortTensor, short);
//                break;
//
//        case Tensor::Int:
//                SW_EVAL_COPY(IntTensor, int);
//                break;
//
//        case Tensor::Long:
//                SW_EVAL_COPY(LongTensor, long);
//                break;
//
//        case Tensor::Float:
//                SW_EVAL_COPY(FloatTensor, float);
//                break;
//
//        case Tensor::Double:
//                SW_EVAL_COPY(DoubleTensor, double);
//                break;
//        }
//}
//
///////////////////////////////////////////////////////////////////////////
//// Scale some input tensor (considered integral image) to the model size
//
//void ipSWEvaluator::iscaleInput(const Tensor& input)
//{
//        switch (input.getDatatype())
//        {
//        case Tensor::Char:
//                SW_EVAL_SCALE(CharTensor, char);
//                break;
//
//        case Tensor::Short:
//                SW_EVAL_SCALE(ShortTensor, short);
//                break;
//
//        case Tensor::Int:
//                SW_EVAL_SCALE(IntTensor, int);
//                break;
//
//        case Tensor::Long:
//                SW_EVAL_SCALE(LongTensor, long);
//                break;
//
//        case Tensor::Float:
//                SW_EVAL_SCALE(FloatTensor, float);
//                break;
//
//        case Tensor::Double:
//                SW_EVAL_SCALE(DoubleTensor, double);
//                break;
//        }
//}
//
///////////////////////////////////////////////////////////////////////////
///// Change the sub-window to process in - overriden
///// Checks also if there is some pattern in this sub-window
//
//bool ipSWEvaluator::setSubWindow(int sw_x, int sw_y, int sw_w, int sw_h)
//{
//        // Set the sub-window
//        const bool changed_size = m_sw_w != sw_w || m_sw_h != sw_h;
//        if (    m_classifier == 0 ||
//                m_buffTensor == 0 ||
//                sw_x < 0 || sw_y < 0 || sw_w <= 0 || sw_h <= 0 ||
//                sw_x + sw_w >= m_input_copy->size(1) ||
//                sw_y + sw_h >= m_input_copy->size(0) ||
//                ipSubWindow::setSubWindow(sw_x, sw_y, sw_w, sw_h) == false)
//        {
//                return false;
//        }
//
//        // If the sub-window size is changed and on the multiscale part
//        //      (the sub-window is different from the classifier's size),
//        //      then the scalling coefficients should be recomputed!
//        if (    changed_size == true &&
//                (sw_w != getModelWidth() || sw_h != getModelHeight()))
//        {
//        	// Cleanup
//                delete[] m_scale_br_indexes;
//                delete[] m_scale_tl_indexes;
//                delete[] m_scale_tr_indexes;
//                delete[] m_scale_bl_indexes;
//                delete[] m_scale_cell_sizes;
//                m_scale_br_indexes = 0;
//                m_scale_tl_indexes = 0;
//                m_scale_tr_indexes = 0;
//                m_scale_bl_indexes = 0;
//                m_scale_cell_sizes = 0;
//
//                // Do the prescalling
//                switch (m_buffTensor->getDatatype())
//                {
//                case Tensor::Char:
//                        SW_EVAL_PRESCALE(CharTensor);
//                        break;
//
//                 case Tensor::Short:
//                        SW_EVAL_PRESCALE(ShortTensor);
//                        break;
//
//                 case Tensor::Int:
//			SW_EVAL_PRESCALE(IntTensor);
//                        break;
//
//                 case Tensor::Long:
//                        SW_EVAL_PRESCALE(LongTensor);
//                        break;
//
//                 case Tensor::Float:
//                        SW_EVAL_PRESCALE(FloatTensor);
//                        break;
//
//                 case Tensor::Double:
//                        SW_EVAL_PRESCALE(DoubleTensor);
//                        break;
//
//                default:
//                        return false;
//                }
//        }
//
//        // If the sub-window has the size of the machine,
//        //      then forward to the classifier the sub-window area of the input
//        if (m_sw_w == getModelWidth() && m_sw_h == getModelHeight())
//        {
//                cropInput(*m_input_copy);
//        }
//
//        // Otherwise, need to rescale the sub-window and forward this to the classifier
//        // (the input tensor is considered and integral image of some features)
//        else
//        {
//                iscaleInput(*m_input_copy);
//        }
//
//        // Just forward the buffer tensor to the classifier
//        const bool processed = m_classifier->forward(*m_buffTensor);
//
//        // Save the buffer tensor if contains a pattern (if requested)
//        if (    m_save_buffTensor == true &&
//                m_buffTensor->nDimension() == 3 &&
//                processed == true &&
//                m_classifier->isPattern() == true)
//        {
//                Image image;
//                if (    image.resize(   m_buffTensor->size(0),
//                                        m_buffTensor->size(1),
//                                        m_buffTensor->size(2)) == true &&
//                        image.copyFrom(*m_buffTensor) == true)
//                {
//                        char str[200];
//                        sprintf(str, "BuffTensor_sw_%d_%d_%dx%d.jpg",
//                                m_sw_x, m_sw_y, m_sw_w, m_sw_h);
//
//                        xtprobeImageFile xtprobe;
//                        xtprobe.save(image, str);
//
//                        /*
//                        // Save also the image as the bindata (with the model confidence in the name)
//                        sprintf(str, "BuffTensor_sw_%d_%d_%dx%d_input_%dx%d_conf_%f.bindata",
//                                m_sw_x, m_sw_y, m_sw_w, m_sw_h,
//                                m_inputSize.w, m_inputSize.h,
//                                m_classifier->getConfidence());
//
//                        const int n_samples = 1;
//                        const int sample_size = m_classifier->getModelWidth() * m_classifier->getModelHeight();
//
//                        File bindata;
//                        if (    bindata.open(str, "w+") == true &&
//                                bindata.write(&n_samples, sizeof(int), 1) == 1 &&
//                                bindata.write(&sample_size, sizeof(int), 1) == 1)
//                        {
//                                ShortTensor* tensor = (ShortTensor*)m_buffTensor;
//                                const float inv_grey = 1.0f / 255.0f;
//
//                                for (int y = 0; y < m_classifier->getModelHeight(); y ++)
//                                        for (int x = 0; x < m_classifier->getModelWidth(); x ++)
//                                        {
//                                                const float value = inv_grey * tensor->get(y, x, 0);
//                                                bindata.write(&value, sizeof(float), 1);
//                                        }
//                        }
//                        */
//                }
//        }
//
//        // OK
//        return processed;
//}
//
///////////////////////////////////////////////////////////////////////////
//
//}
