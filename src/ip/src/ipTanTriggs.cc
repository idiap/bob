#include "ip/ipTanTriggs.h"
#include "ip/ipRescaleGray.h"
#include <cmath>
//STL #include <limits>


/////////////////////////////////////////////////////////////////////////////////////////

namespace Torch {

  /////////////////////////////////////////////////////////////////////////////////////////
  // Constructor
  ipTanTriggs::ipTanTriggs() 
    : 	ipCore()
  {
    addIOption("s_step", 1, "Scale step: (2*s_step)");
    addDOption("gamma", 0.2, "Exponent gamma of the gamma correction");
    addDOption("sigma0", 1.0, "sigma0 of the DoG filter (inner gaussian)");
    addDOption("sigma1", 2.0, "sigma1 of the DoG filter (outer gaussian)");
    addIOption("size", 2, "Subsize of each side of the DoG filter (size=2*subsize+1)");
    addDOption("threshold", 10.0, "Threshold for the contrast equalization");
    addDOption("alpha", 0.1, "Exponent for defining a \"alpha\" norm");
  }

  /////////////////////////////////////////////////////////////////////////////////////////
  // Destructor
  ipTanTriggs::~ipTanTriggs()
  {
  }

  /////////////////////////////////////////////////////////////////////////////////////////
  // Check if the input tensor has the right dimensions and type
  bool ipTanTriggs::checkInput(const Tensor& input) const
  {
    // Accept only 3D tensors of Torch::Image type
    if (    input.nDimension() != 3 ||
        input.getDatatype() != Tensor::Short)
    {
      warning("ipTanTriggs::checkInput(): Incorrect Tensor type and dimension.");
      return false;
    }
    // Accept only gray images
    if (	input.size(2) !=1 )
    {
      warning("ipTanTriggs::checkInput(): Non gray level image (multiple channels).");
      return false;
    }

    // OK
    return true;
  }

  /////////////////////////////////////////////////////////////////////////////////////////
  // Allocate (if needed) the output tensors given the input tensor dimensions
  bool ipTanTriggs::allocateOutput(const Tensor& input)
  {
    if (	m_output == 0 ||
        m_output[0]->nDimension() != 3 ||
        m_output[0]->size(0) != input.size(0) ||
        m_output[0]->size(0) != input.size(1) ||
        m_output[0]->size(0) != input.size(2) )
    {
      cleanup();

      // Need allocation
      m_n_outputs = 1;
      m_output = new Tensor*[m_n_outputs];
      m_output[0] = new ShortTensor(input.size(0), input.size(1), input.size(2));
      return true;
    }

    return true;
  }

  /////////////////////////////////////////////////////////////////////////////////////////
  // Process some input tensor (the input is checked, the outputs are allocated)
  bool ipTanTriggs::processInput(const Tensor& input)
  {
    // Get parameters
    const double sigma0 = getDOption("sigma0");
    const double sigma1 = getDOption("sigma1");
    const int real_size = 2*getIOption("size")+1;
    const double gamma = getDOption("gamma");
    const double alpha = getDOption("alpha");
    const double threshold = getDOption("threshold");

    // Prepare pointers to access pixels
    const ShortTensor* t_input = (ShortTensor*)&input;
    ShortTensor* t_output = (ShortTensor*)m_output[0];

    // An index for the 3D tensor is: [y * stride_h + x * stride_w + p * stride_p]
    const int width = input.size(1);
    const int height = input.size(0);
    const int wxh = width * height;

    // Temporary output destination tensor
    DoubleTensor t_output_double1( t_output->size(0), t_output->size(1), t_output->size(2) );

    //////// PERFORM GAMMA COMPRESSION /////////
    // Initialize indices
    for (int y=0; y<height; y++) 
    {
      for (int x=0; x<width; x++ )
      {	
        if ( fabs(gamma) > 1e-12 ) //STL std::numeric_limits<double>::epsilon( ) )
          (t_output_double1)(y,x,0) = pow( (*t_input)(y,x,0), gamma );
        // TODO which value to add in the log?
        else
          (t_output_double1)(y,x,0) = log( 1. + (*t_input)(y,x,0) );
      }
    }

    //////// PERFORM DoG FILTERING /////////
    // Compute DoG kernel
    DoubleTensor* DoG=computeDoG(sigma0, sigma1, real_size);

    // Temporary output destination tensor
    DoubleTensor t_output_double2( t_output->size(0), t_output->size(1), t_output->size(2) );

    // Perform convolution using mirror interpolation
    const int r=real_size/2;
    for (int y = 0; y < height; y ++)
    {
      for (int x = 0; x < width; x ++)
      {
        // Apply the kernel for the <y, x> pixel
        double sum = 0.0;
        int yyy, xxx;
        for (int yy = -r; yy <= r; yy ++)
        {
          // mirror interpolation
          yyy=yy+y;
          if (yyy<0)
            yyy=abs(yyy)-1;
          if (yyy>=height)
            yyy=2*height-yyy-1;
          for (int xx = -r; xx <= r; xx ++)
          {
            // mirror interpolation
            xxx=xx+x;
            if (xxx<0)
              xxx=abs(xxx)-1;
            if (xxx>=width)
              xxx=2*width-xxx-1;
            sum += 	DoG->get(yy + r, xx + r) * (t_output_double1)(yyy,xxx,0);
          }
        }

        // Save the double value
        (t_output_double2)(y,x,0) = sum;
      }
    }


    //////// PERFORM CONTRAST EQUALIZATION ////////
    double inv_alpha=1./alpha;

    // first step: I:=I/mean(abs(I)^a)^(1/a)
    double sum=0.;
    for (int y = 0; y < height; y ++)
    {
      for (int x = 0; x < width; x ++ )
      {
        sum += pow( fabs((t_output_double2)(y,x,0)), alpha);
      }
    }
    sum = pow( sum/wxh , inv_alpha);

    for (int y = 0; y < height; y ++)
    {
      for (int x = 0; x < width; x ++ )
      {
        (t_output_double2)(y,x,0) /= sum;
      }
    }


    // Second step: I:=I/mean(min(threshold,abs(I))^a)^(1/a)
    double threshold_alpha = pow( threshold, alpha );
    sum=0.;
    for (int y = 0; y < height; y ++)
    {
      for (int x = 0; x < width; x++)
      {
        double var;
        if (  fabs((t_output_double2)(y,x,0))<threshold )
        {
          var =  pow( fabs( (t_output_double2)(y,x,0) ), alpha );
        }
        else
          var = threshold_alpha;
        sum+=var;
      }
    }
    sum = pow( sum/wxh , inv_alpha);

    for (int y = 0; y < height; y ++)
    {
      for (int x = 0; x < width; x ++ )
      {
        (t_output_double2)(y,x,0) /= sum;
      }
    }

    // Last step: I:= threshold * tanh( I / threshold )	
    for (int y = 0; y < height; y ++)
    {
      for (int x = 0; x < width; x ++ )
      {
        (t_output_double2)(y,x,0) = threshold * tanh( (t_output_double2)(y,x,0) / threshold );
      }
    }


    ///////// RESCALE /////////////////	
    // Rescale the values in [0,255] and copy it into the output Tensor
    ipCore *rescale = new ipRescaleGray();
    CHECK_FATAL(rescale->process(t_output_double2) == true);
    const ShortTensor* out = (const ShortTensor*)&rescale->getOutput(0);
    t_output->copy( out );
    delete rescale;

    delete DoG;

    return true;
  }

  DoubleTensor* ipTanTriggs::computeDoG(double sigma0, double sigma1, int size)
  {
    // TODO: Check that size is an odd number
    DoubleTensor* res=new DoubleTensor(size, size);

    const double inv_sigma0_2 = 0.5  / (sigma0*sigma0);
    const double inv_sigma1_2 = 0.5  / (sigma1*sigma1);

    int center=size/2;

    double* g0 = new double[size*size];
    double* g1 = new double[size*size];

    double sum0 = 0.;
    double sum1 = 0.;
    for (int y = 0 ; y < size ; y++)
    {
      for (int x = 0 ; x < size ; x++)
      {
        int yy = y - center;
        int xx = x - center;
        int xx2 = xx*xx;
        int yy2 = yy*yy;

        int ind = y * size + x;

        g0[ ind ] = exp( - inv_sigma0_2 * (xx2 + yy2) );
        g1[ ind ] = exp( - inv_sigma1_2 * (xx2 + yy2) );

        sum0 += g0[ind];
        sum1 += g1[ind];			
      }
    }

    // Normalize the kernel such that the sum over the area is equal to 1
    const double inv_sum0 = 1.0 / sum0;
    const double inv_sum1 = 1.0 / sum1;

    for (int y = 0 ; y < size ; y++)
    {
      for (int x = 0 ; x < size ; x++ )
      {
        int ind =  y * size + x;
        (*res)(y,x) = inv_sum0 * g0[ ind ] - inv_sum1 * g1[ ind ];
      }
    }

    delete[] g0;
    delete[] g1;

    return res;
  }


}

