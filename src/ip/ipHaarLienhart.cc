#include "ipHaarLienhart.h"

namespace Torch
{

/////////////////////////////////////////////////////////////////////////
// Constructor

    ipHaarLienhart::ipHaarLienhart() //int width_, int height_)
            :	ipCore()
    {

        nparams =4; // number of parameters for each rectangle x,y,w,h,weight
        t_ = new DoubleTensor();
        t__ = new DoubleTensor();
        parameters=NULL;
        weight =NULL;
        u_x = -1;
        u_y= -1;
        u_z =-1;
        u_size_x  = -1;
        u_size_y =-1;
        u_size_z  = -1;
        u_parameters = NULL;
        u_weight = NULL;

        //print("ipHaar() Type-%d (%d-%d) [%dx%d]\n", type, x, y, w, h);


    }

/////////////////////////////////////////////////////////////////////////
// Destructor

    ipHaarLienhart::~ipHaarLienhart()
    {
        delete parameters;
        delete weight;
        delete t_;
        delete t__;
        delete u_parameters;
        delete u_weight;
    }

///////////////////////////////////////////////////////////////////////////////////////////////////
// Check if the input tensor has the right dimensions and type - overriden
    bool ipHaarLienhart::setNoRec(int noRecs_)
    {
        noRecs = noRecs_;
        parameters = new int[noRecs*nparams];
        u_parameters = new int[noRecs*nparams];
        u_weight = new double[noRecs];
        weight = new double[noRecs];



        return true;
    }
///////////////////////////////////////////////////////////////////////////////////////////
    bool ipHaarLienhart::setRec(int Rec_,double weight_, int x_, int y_, int w_, int h_)
    {
        //for(int i=0;i<5;i++)
        if (Rec_>noRecs-1)
        {
            Torch::error("ipHaarLienhart::setRec() The rec parameters exceeds the number of Recs set");
            return false;
        }
        int k = Rec_*nparams;
        parameters[k+0] = x_;
        parameters[k+1] = y_;
        parameters[k+2] = w_;
        parameters[k+3] = h_;
        //parameters[k+4] = weight_;
        weight[Rec_] = weight_;

        u_parameters[k+0] = x_;
        u_parameters[k+1] = y_;
        u_parameters[k+2] = w_;
        u_parameters[k+3] = h_;
        //parameters[k+4] = weight_;
        u_weight[Rec_] = weight_;

//        for(int i=0;i<5;i++)
//            print("feature i %d\n",parameters[k+i]);
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
      //  print("tndimension %d\n",tndimension);
        if ( !(tndimension ==2 || tndimension ==3))
        {
            print("Warning....... just Implemented for 2D and 3D.............\n");
        }

        if (tndimension==2)
        {
            u_x= m_region.pos[0];
            u_y = m_region.pos[1];
            // u_size_x = m_region.size[1];
            // u_size_y = m_region.size[0];
            if ( u_size_x != m_region.size[0] || u_size_y != m_region.size[1])
            {
                updateParameters();
                u_size_y = m_region.size[1];
                u_size_x = m_region.size[0];
            }
        }
        //....for 3 dimension data
        if (tndimension==3)
        {
            u_x= m_region.pos[0];
            u_y = m_region.pos[1];
            u_z = m_region.pos[2]; //this gives the plane to operate on

            if (m_region.size[2] != 1)
            {
                print("Error ........ the thirsd dimension cannot be  more that size 1.....\n");
            }
            else
                u_size_z = m_region.size[2];


            if ( u_size_x != m_region.size[0] || u_size_y != m_region.size[1])
            {
                //update the parameters
                updateParameters();
                u_size_x = m_region.size[0];
                u_size_y = m_region.size[1];
                u_size_z = m_region.size[2]; //can be used for 3D haar - but now it is 1
            }

        }
      //  print("............iphaar. you are here\n");
    }
    ////////////////////////////////////////////////////////////////////////////////////
    void ipHaarLienhart::updateParameters()
    {
        //only the parameters has to be updated.
        //have to find the scale and update the parameters
        double sW,sH;
        sW = (double)((u_size_x+0.0)/(width+0.0));
        sH = (double)((u_size_y+0.0)/(height+0.0));
        //you have to loop for the number of rectangles
        // you have to first find the relative position for new width and height
        // and then see that w and h of iphaar is also updated
        //you will have to see that the area is same for +ve and negative area
        //the weights have to be adjusted to compensate for the scaling errors
        int k;
        int *A = new int[noRecs];
        int AP; //+ve area
        double WN,WP; //sum of +ve and -ve weights
        int AN; //-ve area
        int nc=0;
        int pc = 0;
        AP=0;
        AN=0;
        WN=0;
        WP=0;
        for (int i=0;i<noRecs;i++)
        {
            k =i*nparams;
            u_parameters[k+0] = int(sW*(parameters[k+0]));//+0.5));
//            if (u_parameters[k+0] != parameters[k+0])
//                print("............they are not the same....\n");
            u_parameters[k+1] = int(sH*(parameters[k+1]));//+0.5));
            u_parameters[k+2] = int(sW*(parameters[k+2]));//+0.5));
            u_parameters[k+3] = int(sH*(parameters[k+3]));//+0.5));
            A[i] = u_parameters[k+2] * u_parameters[k+3];
            if (weight[i] >0)
            {
                pc++;
                AP +=A[i];
                WP += weight[i];
            }

            else
            {
                nc++;
                AN += A[i];
                WN += weight[i];
            }

        }

        //now update the weights too
        //keep the positive weights same and change the -ve weights
        double wr;
        wr = WP*(AP+0.0)/(AN+0.0);
        for (int i=0;i<noRecs;i++)
        {
            if (weight[i]<0)
                u_weight[i] = weight[i]*wr;
            else
                u_weight[i] = weight[i];
//            if (u_weight[i] != weight[i])
//                print(",.............not same\n");
        }
        delete A;

    }
///////////////////////////////////////////////////////////////////////////////////////////
    void	ipHaarLienhart::setModelSize(const TensorSize& modelSize)
    {

        m_modelSize = modelSize;
        height = m_modelSize.size[1];
        width = m_modelSize.size[0];
        u_size_x = width;
        u_size_y =  height;
        u_size_z =1;
        u_z=0;
        u_x = 0;
        u_y=0;
      //  print("............iphaar. you are here\n");

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
        const DoubleTensor* t_input = (DoubleTensor*)&input;
        DoubleTensor* t_output = (DoubleTensor*)m_output[0];


        double sum=0;
        int k;
        int *t1 = new int[4];
        int tensor_width, tensor_height;

if(u_x<0 || u_y <0)
 print("Ux,Uy : %d,%d\n",u_x,u_y);
        if (t_input->nDimension()==2)
        {
            tensor_height = t_input->size(1);
            tensor_width = t_input->size(0);
            for (int i=0;i<noRecs;i++)
            {
                k=i*nparams;
///.........
                // if ( !( (u_y+u_parameters[k+0] + u_parameters[k+2]) <tensor_height || (u_x+u_parameters[k+1]+u_parameters[k+3])< tensor_width))
                if ( !( (u_x+u_parameters[k+0] + u_parameters[k+2]) <tensor_width && (u_y+u_parameters[k+1]+u_parameters[k+3])< tensor_height))
                {
                    print("Error .ipHaarLienhart out of range\n");
                    return false;
                }

//top left
                if ( (u_x+u_parameters[k+0]-1) < 0 || (u_y+u_parameters[k+1]-1)<0)
                    t1[0]=0;
                else
                    //  t1[0] = t_input->get(u_y+u_parameters[k+0]-1,u_x+u_parameters[k+1]-1);
                    t1[0] = t_input->get(u_x+u_parameters[k+0]-1,u_y+u_parameters[k+1]-1);



//bottom right
                if ( (u_x+u_parameters[k+0]+u_parameters[k+2]-1)<0 || (u_y+u_parameters[k+1]+u_parameters[k+3]-1)<0)
                    t1[1]=0;
                else
                    //    t1[1] = t_input->get(u_y+u_parameters[k+0]+u_parameters[k+2]-1,u_x+u_parameters[k+1]+u_parameters[k+3]-1);
                    t1[1] = t_input->get(u_x+u_parameters[k+0]+u_parameters[k+2]-1,u_y+u_parameters[k+1]+u_parameters[k+3]-1);


//top right
                if ( (u_x+u_parameters[k+0]+u_parameters[k+2]-1) <0 || (u_y+u_parameters[k+1]-1) <0)
                    t1[2]=0;
                else
                    //   t1[2] = t_input->get(u_y+u_parameters[k+0]+u_parameters[k+2]-1,u_x+u_parameters[k+1]-1);
                    t1[2] = t_input->get(u_x+u_parameters[k+0]+u_parameters[k+2]-1,u_y+u_parameters[k+1]-1);

//bottom left
                if ( (u_x+u_parameters[k+0]-1 <0) || (u_y+u_parameters[k+1]+u_parameters[k+3]-1) <0 )
                    t1[3]=0;
                else
                    // t1[3] = t_input->get(u_y+u_parameters[k+0]-1,u_x+u_parameters[k+1]+u_parameters[k+3]-1);
                    t1[3] = t_input->get(u_x+u_parameters[k+0]-1,u_y+u_parameters[k+1]+u_parameters[k+3]-1);


                sum += (t1[0]+t1[1]-t1[2]-t1[3]) *  u_weight[i];
            }
            (*t_output)(0) =sum;
           // print("Feature value inside %f\n",sum);
        }

        //.....for 3D data, have to handle the plane in which it operates
        if (t_input->nDimension()==3)
        {
            int tensor_plane;
            tensor_height = t_input->size(1);
            tensor_width = t_input->size(0);
            tensor_plane = t_input->size(2);

            if ( !(u_z<tensor_plane && u_z>=0))
            {
                print("Error: ipHaarLienhart plane out of range\n");
            }

            for (int i=0;i<noRecs;i++)
            {
                k=i*nparams;
///.........
                if ( !( (u_y+u_parameters[k+0] + u_parameters[k+2]) <tensor_height || (u_x+u_parameters[k+1]+u_parameters[k+3])< tensor_width))

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
            (*t_output)(0) =sum;
        }
        delete t1;

        //  print(" Sum %f\n",sum);
        return true;
    }
/////////////////////////////////////////////////////////////////////////////////////////////////////
    bool ipHaarLienhart::loadFile(File& file)
    {


        if (file.taggedRead(&width, sizeof(int), 1, "Width") != 1)
        {
            Torch::message("ipHaarLienhart::load - failed to read <Width> field!\n");
            return false;
        }


        if (file.taggedRead(&height, sizeof(int), 1, "Height") != 1)
        {
            Torch::message("ipHaarLienhart::load - failed to read <Height> field!\n");
            return false;
        }

        m_modelSize.size[1] = height;
        m_modelSize.size[0]=width;

        if (file.taggedRead(&noRecs, sizeof(int), 1, "NoRecs") != 1)
        {
            Torch::message("ipHaarLienhart::load - failed to read <NoRecs> field!\n");
            return false;
        }

        int nsize = noRecs*nparams;
        parameters = new int[nsize];

        if (file.taggedRead(parameters, sizeof(int),nsize , "parameters") != nsize)
        {
            Torch::message("ipHaarLienhart::load - failed to read <parameters> field!\n");
            return false;
        }


        weight = new double[noRecs];
        if (file.taggedRead(weight, sizeof(double),noRecs , "weight") != noRecs)
        {
            Torch::message("ipHaarLienhart::load - failed to read <weight> field!\n");
            return false;
        }


        u_parameters = new int[nsize];
        u_weight = new double[noRecs];
        //initialize with the same parameters
        for (int i=0;i<nsize;i++)
            u_parameters[i] = parameters[i];
        for (int i=0;i<noRecs;i++)
            u_weight[i] = weight[i];
        u_size_x = width;
        u_size_y= height;
        u_x =0;
        u_y=0;
        //.....here call setregion with 0,0,w,h
        TensorRegion *temp_region= new TensorRegion(0,0,width,height);

        setRegion(*temp_region);

        print("ipHaarLienhart()::loadFile()\n");
        print("   Number of Rectangles = %d\n", noRecs);


        return true;

    }

    bool ipHaarLienhart::saveFile(File& file) const
    {
        int idCore =2;
        if (file.taggedWrite(&idCore, sizeof(int), 1, "CoreID") != 1)
        {
            Torch::message("ipHaarLienhart::save - failed to write <CoreID> field!\n");
            return false;
        }


        //  m_modelSize[0]
        if (file.taggedWrite(&width, sizeof(int), 1, "Width") != 1)
        {
            Torch::message("ipHaarLienhart::save - failed to write <Width> field!\n");
            return false;
        }

        if (file.taggedWrite(&height, sizeof(int), 1, "Height") != 1)
        {
            Torch::message("ipHaarLienhart::save - failed to write <height> field!\n");
            return false;
        }

        if (file.taggedWrite(&noRecs, sizeof(int), 1, "NoRecs") != 1)
        {
            Torch::message("ipHaarLienhart::save - failed to write <NoRecs> field!\n");
            return false;
        }

        int nsize = noRecs*nparams;
        if (file.taggedWrite(parameters, sizeof(int), nsize, "parameters") != nsize)
        {
            Torch::message("ipHaarLienhart::save - failed to write <parameters> field!\n");
            return false;
        }

        if (file.taggedWrite(weight, sizeof(double), noRecs, "weight") != noRecs)
        {
            Torch::message("ipHaarLienhart::save - failed to write <weight> field!\n");
            return false;
        }


        print("ipHaarLienhart()::saveFile()\n");
        print("   Number of Rectangles = %d\n", noRecs);
        // print("   X-Y = (%d-%d)\n", x, y);
        // print("   WxH = [%dx%d]\n", w, h);


        return true;
    }

/////////////////////////////////////////////////////////////////////////

}
