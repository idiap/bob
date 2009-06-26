#include "ipHaarLienhart.h"

namespace Torch
{

/////////////////////////////////////////////////////////////////////////
// Constructor

    ipHaarLienhart::ipHaarLienhart() //int width_, int height_)
            :	ipCore()
    {

        m_nparams =4; // number of parameters for each rectangle x,y,w,h,weight
        //   t_ = new DoubleTensor();
        //  t__ = new DoubleTensor();
        m_parameters=NULL;
        m_weight =NULL;
        u_x = -1;
        u_y= -1;
        u_z =-1;
        u_size_x  = -1;
        u_size_y =-1;
        u_size_z  = -1;
        u_parameters = NULL;
        u_weight = NULL;
        invscale=1;

        //print("ipHaar() Type-%d (%d-%d) [%dx%d]\n", type, x, y, w, h);


    }

/////////////////////////////////////////////////////////////////////////
// Destructor

    ipHaarLienhart::~ipHaarLienhart()
    {
        delete[] m_parameters;
        delete[] m_weight;
        //  delete t_;
        // delete t__;
        delete[] u_parameters;
        delete[] u_weight;
    }

///////////////////////////////////////////////////////////////////////////////////////////////////
// Check if the input tensor has the right dimensions and type - overriden
    bool ipHaarLienhart::setNoRec(int noRecs_)
    {
        m_noRecs = noRecs_;
        m_parameters = new int[m_noRecs*m_nparams];
        u_parameters = new int[m_noRecs*m_nparams];
        u_weight = new double[m_noRecs];
        m_weight = new double[m_noRecs];



        return true;
    }
///////////////////////////////////////////////////////////////////////////////////////////
    bool ipHaarLienhart::setRec(int Rec_,double weight_, int y_, int x_, int h_, int w_)
    {
        //for(int i=0;i<5;i++)
        if (Rec_>m_noRecs-1)
        {
            Torch::error("ipHaarLienhart::setRec() The rec parameters exceeds the number of Recs set");
            return false;
        }
        int k = Rec_*m_nparams;
        m_parameters[k+0] = y_;
        m_parameters[k+1] = x_;
        m_parameters[k+2] = h_;
        m_parameters[k+3] = w_;
        //parameters[k+4] = weight_;
        m_weight[Rec_] = weight_;

        u_parameters[k+0] = y_;
        u_parameters[k+1] = x_;
        u_parameters[k+2] = h_;
        u_parameters[k+3] = w_;
        //parameters[k+4] = weight_;
        u_weight[Rec_] = weight_;


        return true;

    }
//////////////////////////////////////////////////////////////////////////////
    void	ipHaarLienhart::setRegion(const TensorRegion& region)
    {
        //there are two things that needs to be done
        //1. if the region.size == previous region.size
        m_region = region;
        int tndimension; //temporary tracking of n_dimension
        tndimension= m_region.n_dimensions;

        if ( !(tndimension ==2 || tndimension ==3))
        {
            print("Warning....... just Implemented for 2D and 3D.............\n");
        }

        if (tndimension==2)
        {
            u_x= m_region.pos[1];
            u_y = m_region.pos[0];
            u_z= 0;
            // u_size_x = m_region.size[1];
            // u_size_y = m_region.size[0];
            if ( u_size_x != m_region.size[1] || u_size_y != m_region.size[0])
            {
                u_size_y = m_region.size[0];
                u_size_x = m_region.size[1];
                updateParameters();
            }
        }
        //....for 3 dimension data
        if (tndimension==3)
        {
            u_x= m_region.pos[1];
            u_y = m_region.pos[0];
            u_z = m_region.pos[2]; //this gives the plane to operate on

            if (m_region.size[2] != 1)
            {
                print("Error ........ the thirsd dimension cannot be  more that size 1.....\n");
            }
            else
                u_size_z = m_region.size[2];


            if ( u_size_x != m_region.size[1] || u_size_y != m_region.size[0])
            {
                //update the parameters
                u_size_x = m_region.size[1];
                u_size_y = m_region.size[0];
                u_size_z = m_region.size[2]; //can be used for 3D haar - but now it is 1
                updateParameters();
            }

        }

    }
////////////////////////////////////////////////////////////////////////////////////
    void ipHaarLienhart::updateParameters()
    {
        //only the parameters has to be updated.
        //have to find the scale and update the parameters
        double sW,sH;
        sW = (double)((u_size_x+0.0)/(m_width+0.0));
        sH = (double)((u_size_y+0.0)/(m_height+0.0));

        /// you have to loop for the number of rectangles
        /// you have to first find the relative position for new width and height
        /// and then see that w and h of iphaar is also updated
        /// you will have to see that the area is same for +ve and negative area
        /// the weights have to be adjusted to compensate for the scaling errors
        int k;
        int *A = new int[m_noRecs];
        int AP; //+ve area
        double WN,WP; //sum of +ve and -ve weights
        int AN; //-ve area
        int nc=0;
        int pc = 0;
        AP=0;
        AN=0;
        WN=0;
        WP=0;
        for (int i=0;i<m_noRecs;i++)
        {
            k =i*m_nparams;
            u_parameters[k+0] = int(sH*(m_parameters[k+0]+0.0));//+0.5));
            u_parameters[k+1] = int(sW*(m_parameters[k+1]+0.0));//+0.5));
            u_parameters[k+2] = int(sH*(m_parameters[k+2]+0.0));//+0.5));
            u_parameters[k+3] = int(sW*(m_parameters[k+3]+0.0));//+0.5));
            A[i] = u_parameters[k+2] * u_parameters[k+3];
            if (m_weight[i] >0)
            {
                pc++;
                AP +=A[i];
                WP += abs(m_weight[i]);
            }

            else
            {
                nc++;
                AN += A[i];
                WN += abs(m_weight[i]);
            }

        }

       //now update the weights too
        //keep the positive weights same and change the -ve weights
        double wr;
        wr = WP/WN*(AP+0.0)/(AN+0.0);
        for (int i=0;i<m_noRecs;i++)
        {
            if (m_weight[i]<0)
                u_weight[i] = m_weight[i]*wr;
            else
                u_weight[i] = m_weight[i];

        }

        //........................check the sum is 0.......................................
//        double tempsum  =0;
//        double tempsum2 =0;
//        for (int i=0;i<m_noRecs;i++)
//        {
//            k =i*m_nparams;
//            tempsum = tempsum + u_weight[i]* u_parameters[k+2] *u_parameters[k+3];
//            tempsum2 = tempsum2 + m_weight[i]* m_parameters[k+2] *m_parameters[k+3];
//        }
//        //  print("Sum of area is %f\n",tempsum);
//        if (abs(tempsum)!= 0.000)
//            print("Sum of area is %f - number of rectangles %d Original value %f\n",tempsum, m_noRecs,tempsum2);

        //......................................................................................

        invscale = sW*sH;
        delete [] A;

    }
///////////////////////////////////////////////////////////////////////////////////////////
    void  ipHaarLienhart::setModelSize(const TensorSize& modelSize)
    {

        m_modelSize = modelSize;
        m_height = m_modelSize.size[0];
        m_width = m_modelSize.size[1];
        u_size_x = m_width;
        u_size_y =  m_height;
        u_size_z =1;
        u_z=0;
        u_x = 0;
        u_y=0;


    }

////////////////////////////////////////////////////////////////////////////////////////
    bool ipHaarLienhart::checkInput(const Tensor& input) const
    {
        if ( !(input.nDimension() == 2 ||  input.nDimension()==3))
        {
            Torch::error("ipHaarLienhart::checkInput() input should be 2D or 3D");
            return false;
        }

        //u can even check if the width and height are 0

        // OK
        return true;
    }

/////////////////////////////////////////////////////////////////////////
// Allocate (if needed) the output tensors given the input tensor dimensions

    bool ipHaarLienhart::allocateOutput(const Tensor& input)
    {
        // Allocate the output if needed
        if (m_output == 0)
        {
            m_n_outputs = 1;
            m_output = new Tensor*[m_n_outputs];
            m_output[0] = new DoubleTensor(1);
        }
        return true;
    }
///////////////////////////////////////////////////////////////////////////////////////////////
    bool ipHaarLienhart::processInput(const Tensor& input)
    {

        //whatever has to be processed here should use u_parameters, u_x, u_y, u_weight, u_z, u_size_x, u_size_y, u_size_z
        const IntTensor* t_input = (IntTensor*)&input;
        DoubleTensor* t_output = (DoubleTensor*)m_output[0];

        double sum=0;
        int k;
        int t1[4];
        int tensor_width, tensor_height;

        if (u_x<0 || u_y <0)
            print("Ux,Uy : %d,%d\n",u_x,u_y);
        if (t_input->nDimension()==2)
        {
            tensor_height = t_input->size(0);
            tensor_width = t_input->size(1);
            for (int i=0;i<m_noRecs;i++)
            {
                k=i*m_nparams;
///.........
                // if ( !( (u_y+u_parameters[k+0] + u_parameters[k+2]) <tensor_height || (u_x+u_parameters[k+1]+u_parameters[k+3])< tensor_width))
                if ( !( (u_x+u_parameters[k+1] + u_parameters[k+3]) <tensor_width && (u_y+u_parameters[k+0]+u_parameters[k+2])< tensor_height))
                {
                    print("Error .ipHaarLienhart out of range\n");
                    return false;
                }

//top left
                if ( (u_x+u_parameters[k+1]-1) < 0 || (u_y+u_parameters[k+0]-1)<0)
                    t1[0]=0;
                else
                    //  t1[0] = t_input->get(u_y+u_parameters[k+0]-1,u_x+u_parameters[k+1]-1);
                    t1[0] = t_input->get(u_y+u_parameters[k+0]-1,u_x+u_parameters[k+1]-1);



//bottom right
                if ( (u_x+u_parameters[k+1]+u_parameters[k+3]-1)<0 || (u_y+u_parameters[k+0]+u_parameters[k+2]-1)<0)
                    t1[1]=0;
                else
                    //    t1[1] = t_input->get(u_y+u_parameters[k+0]+u_parameters[k+2]-1,u_x+u_parameters[k+1]+u_parameters[k+3]-1);
                    t1[1] = t_input->get(u_y+u_parameters[k+0]+u_parameters[k+2]-1, u_x+u_parameters[k+1]+u_parameters[k+3]-1);


//top right
                if ( (u_x+u_parameters[k+1]+u_parameters[k+3]-1) <0 || (u_y+u_parameters[k+0]-1) <0)
                    t1[2]=0;
                else
                    //   t1[2] = t_input->get(u_y+u_parameters[k+0]+u_parameters[k+2]-1,u_x+u_parameters[k+1]-1);
                    t1[2] = t_input->get(u_y+u_parameters[k+0]-1,u_x+u_parameters[k+1]+u_parameters[k+3]-1);

//bottom left
                if ( (u_x+u_parameters[k+1]-1 <0) || (u_y+u_parameters[k+0]+u_parameters[k+2]-1) <0 )
                    t1[3]=0;
                else
                    // t1[3] = t_input->get(u_y+u_parameters[k+0]-1,u_x+u_parameters[k+1]+u_parameters[k+3]-1);
                    t1[3] = t_input->get(u_y+u_parameters[k+0]+u_parameters[k+2]-1,u_x+u_parameters[k+1]-1);


                sum += (t1[0]+t1[1]-t1[2]-t1[3]) *  u_weight[i];
            }
            (*t_output)(0) =sum/invscale;
            // print("Feature value inside %f\n",sum);
        }

        //.....for 3D data, have to handle the plane in which it operates
        if (t_input->nDimension()==3)
        {
            int tensor_plane;
            tensor_height = t_input->size(0);
            tensor_width = t_input->size(1);
            tensor_plane = t_input->size(2);

            if ( !(u_z<tensor_plane && u_z>=0))
            {
                print("Error: ipHaarLienhart plane out of range\n");
            }

            for (int i=0;i<m_noRecs;i++)
            {

                k=i*m_nparams;

///.........
                if ( !( (u_y+u_parameters[k+0] + u_parameters[k+2]) <tensor_height &&
                        (u_x+u_parameters[k+1]+u_parameters[k+3])< tensor_width))

                {
                    print("Error .ipHaarLienhart out of range\n");
                    return false;
                }

//top left
                if ( (u_y+u_parameters[k+0]-1) < 0 || (u_x+u_parameters[k+1]-1)<0)
                    t1[0]=0;
                else
                    t1[0] = t_input->get(u_y+u_parameters[k+0]-1,u_x+u_parameters[k+1]-1,u_z);

//bottom right
                if ( (u_y+u_parameters[k+0]+u_parameters[k+2]-1)<0 || (u_x+u_parameters[k+1]+u_parameters[k+3]-1)<0)
                    t1[1]=0;
                else
                    t1[1] = t_input->get(u_y+u_parameters[k+0]+u_parameters[k+2]-1,u_x+u_parameters[k+1]+u_parameters[k+3]-1,u_z);

//bottom left
                if ( (u_y+u_parameters[k+0]+u_parameters[k+2]-1) <0 || (u_x+u_parameters[k+1]-1) <0)
                    t1[2]=0;
                else
                    t1[2] = t_input->get(u_y+u_parameters[k+0]+u_parameters[k+2]-1,u_x+u_parameters[k+1]-1,u_z);

//top right
                if ( (u_y+u_parameters[k+0]-1 <0) || (u_x+u_parameters[k+1]+u_parameters[k+3]-1) <0 )
                    t1[3]=0;
                else
                    t1[3] = t_input->get(u_y+u_parameters[k+0]-1,u_x+u_parameters[k+1]+u_parameters[k+3]-1,u_z);

                sum += (t1[0]+t1[1]-t1[2]-t1[3]) *  u_weight[i];
            }
            (*t_output)(0) =sum/invscale;
        }
        //delete [] t1;

        //  print(" Sum %f\n",sum);
        return true;
    }
/////////////////////////////////////////////////////////////////////////////////////////////////////
    bool ipHaarLienhart::loadFile(File& file)
    {


        if (file.taggedRead(&m_width, sizeof(int), 1, "Width") != 1)
        {
            Torch::message("ipHaarLienhart::load - failed to read <Width> field!\n");
            return false;
        }


        if (file.taggedRead(&m_height, sizeof(int), 1, "Height") != 1)
        {
            Torch::message("ipHaarLienhart::load - failed to read <Height> field!\n");
            return false;
        }

        m_modelSize.size[0] = m_height;
        m_modelSize.size[1]= m_width;

        if (file.taggedRead(&m_noRecs, sizeof(int), 1, "NoRecs") != 1)
        {
            Torch::message("ipHaarLienhart::load - failed to read <NoRecs> field!\n");
            return false;
        }

        int nsize = m_noRecs*m_nparams;
        delete [] m_parameters;
        m_parameters = new int[nsize];

        if (file.taggedRead(m_parameters, sizeof(int),nsize , "parameters") != nsize)
        {
            Torch::message("ipHaarLienhart::load - failed to read <parameters> field!\n");
            return false;
        }


        m_weight = new double[m_noRecs];
        if (file.taggedRead(m_weight, sizeof(double),m_noRecs , "weight") != m_noRecs)
        {
            Torch::message("ipHaarLienhart::load - failed to read <weight> field!\n");
            return false;
        }


        u_parameters = new int[nsize];
        u_weight = new double[m_noRecs];
        //initialize with the same parameters
        for (int i=0;i<nsize;i++)
            u_parameters[i] = m_parameters[i];
        for (int i=0;i<m_noRecs;i++)
            u_weight[i] = m_weight[i];
        u_size_x = m_width;
        u_size_y= m_height;
        u_x =0;
        u_y=0;
        u_z=0;
        //.....here call setregion with 0,0,h,w
        setRegion(TensorRegion(0,0,m_height,m_width));




        return true;

    }
//////////////////////////////////////////////////////////////////////////////////////
    bool ipHaarLienhart::saveFile(File& file) const
    {
        int idCore = getID();
        if (file.taggedWrite(&idCore, sizeof(int), 1, "CoreID") != 1)
        {
            Torch::message("ipHaarLienhart::save - failed to write <CoreID> field!\n");
            return false;
        }


        //  m_modelSize[0]
        if (file.taggedWrite(&m_width, sizeof(int), 1, "Width") != 1)
        {
            Torch::message("ipHaarLienhart::save - failed to write <Width> field!\n");
            return false;
        }

        if (file.taggedWrite(&m_height, sizeof(int), 1, "Height") != 1)
        {
            Torch::message("ipHaarLienhart::save - failed to write <height> field!\n");
            return false;
        }

        if (file.taggedWrite(&m_noRecs, sizeof(int), 1, "NoRecs") != 1)
        {
            Torch::message("ipHaarLienhart::save - failed to write <NoRecs> field!\n");
            return false;
        }

        int nsize = m_noRecs*m_nparams;
        if (file.taggedWrite(m_parameters, sizeof(int), nsize, "parameters") != nsize)
        {
            Torch::message("ipHaarLienhart::save - failed to write <parameters> field!\n");
            return false;
        }

        if (file.taggedWrite(m_weight, sizeof(double), m_noRecs, "weight") != m_noRecs)
        {
            Torch::message("ipHaarLienhart::save - failed to write <weight> field!\n");
            return false;
        }



        return true;
    }

/////////////////////////////////////////////////////////////////////////

}
