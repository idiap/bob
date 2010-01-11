#include "torch5spro.h"

using namespace Torch;


int main(int argc, char* argv[])
{
    ///////////////////////////////////////////////////////////////////
    // Parse the command line
    ///////////////////////////////////////////////////////////////////

    // Set options
    char* tensor_filename;
    bool verbose;
    bool b_avg;
    bool b_nan;
    bool b_range;
    int dimension,d1,d2,d3,d4;
    int Sstart_range, Send_range;
    Tensor::Type mtype;
    int m_n_samples;

    CmdLine cmd;
    cmd.setBOption("write log", false);

    cmd.info("Tensor Check  - checks for out of range values, creates average, ");

    cmd.addText("\nArguments:");
    cmd.addSCmdArg("tensor file to test", &tensor_filename, "tensor file to check");

    cmd.addText("\nOptions:");
    cmd.addBCmdOption("-verbose", &verbose, false, "print Tensor values");
    cmd.addBCmdOption("-avg", &b_avg, false, "create average value");
    cmd.addBCmdOption("-nan", &b_nan, false, "check for NaN values");
    cmd.addBCmdOption("-range", &b_range, false, "check id all the values are within the range");
    cmd.addICmdOption("-sr", &Sstart_range, 0, "input the range values - start");
    cmd.addICmdOption("-er", &Send_range, 0, "input the range values - end");

    // Parse the command line
    if (cmd.read(argc, argv) < 0)
    {
        return 0;
    }

    FloatTensor *avgTensor;
    TensorFile tf;

    if (b_avg || b_nan || b_range)
    {


        CHECK_FATAL(tf.openRead(tensor_filename));

        print("Reading tensor header file ...\n");
        const TensorFile::Header& header = tf.getHeader();

        print("Tensor file:\n");
        print(" type:         [%s]\n", str_TensorTypeName[header.m_type]);
        print(" n_tensors:    [%d]\n", header.m_n_samples);
        print(" n_dimensions: [%d]\n", header.m_n_dimensions);
        print(" size[0]:      [%d]\n", header.m_size[0]);
        print(" size[1]:      [%d]\n", header.m_size[1]);
        print(" size[2]:      [%d]\n", header.m_size[2]);
        print(" size[3]:      [%d]\n", header.m_size[3]);
        dimension = header.m_n_dimensions;
        m_n_samples = header.m_n_samples;
        mtype = header.m_type;
        d1 = header.m_size[0];
        d2 = header.m_size[1];
        d3 = header.m_size[2];
        d4 = header.m_size[3];

        if (header.m_type != Tensor::Short && header.m_type != Tensor::Int)
        {
            warning("Unsupported tensor type (Short and Int only).");

            return 1;
        }

//Tensor *tensor;
        if (b_avg)
        {
            if (dimension==1)
                avgTensor = new FloatTensor(d1);
            if (dimension==2)
                avgTensor = new FloatTensor(d1,d2);
            if (dimension==3)
                avgTensor = new FloatTensor(d1,d2,d3);
            if (dimension==4)
                avgTensor = new FloatTensor(d1,d2,d3,d4);


            avgTensor->fill(0);
        }


        for (int msamples = 0;msamples< header.m_n_samples;msamples++)
        {
            ShortTensor *Stensor = (ShortTensor *)tf.load();
            if (dimension==1)
            {
                for (int i=0;i<d1;i++)
                {
                    if (b_nan)
                        if (isnan((*Stensor)(i)))
                            print("Nan Value in sample %d\n",msamples);

                    if (b_range)
                        if ( (*Stensor)(i)>Send_range || (*Stensor)(i)<Sstart_range)
                            print("The values are not in range in sample %d\n",msamples);

                    if (b_avg)
                    {
                        (*avgTensor)(i) += (float)(*Stensor)(i);
                    }


                }
            }
            if (dimension==2)
            {
                for (int i=0;i<d1;i++)
                    for (int j=0;j<d2;j++)
                    {
                        if (b_nan)
                            if (isnan((*Stensor)(i,j)))
                                print("Nan Value in sample %d\n",msamples);

                        if (b_range)
                            if ( (*Stensor)(i,j)>Send_range ||(*Stensor)(i,j)<Sstart_range)
                                print("The values are not in range in sample %d\n",msamples);

                        if (b_avg)
                        {
                            (*avgTensor)(i,j) += (float)(*Stensor)(i,j);
                        }

                    }


            }
            if (dimension==3)
            {
                for (int i=0;i<d1;i++)
                    for (int j=0;j<d2;j++)
                        for (int m=0;m<d3;m++)
                        {

                            if (b_nan)
                                if (isnan((*Stensor)(i,j,m)))
                                    print("Nan Value in sample %d\n",msamples);

                            if (b_range)
                                if ( (*Stensor)(i,j,m)>Send_range || (*Stensor)(i,j,m)<Sstart_range)
                                    print("The values are not in range in sample %d\n",msamples);

                            if (b_avg)
                            {
                                (*avgTensor)(i,j,m) += (float)(*Stensor)(i,j,m);
                            }
                        }
            }

            if (dimension==4)
            {
                for (int i=0;i<d1;i++)
                    for (int j=0;j<d2;j++)
                        for (int m=0;m<d3;m++)
                            for (int n=0;n<d4;n++)

                            {
                                if (b_nan)
                                    if (isnan((*Stensor)(i,j,m,n)))
                                        print("Nan Value in sample %d\n",msamples);

                                if (b_range)
                                    if ( (*Stensor)(i,j,m,n)>Send_range || (*Stensor)(i,j,m,n)<Sstart_range)
                                        print("The values are not in range in sample %d\n",msamples);

                                if (b_avg)
                                {
                                    (*avgTensor)(i,j,m,n) += (float)(*Stensor)(i,j,m,n);
                                }
                            }
            }
        }

    }// if b_nan || ..
    if (b_avg)
    {
        TensorFile tfa;
        if (dimension==1)
            if (tfa.openWrite("avgTensor.tensor",mtype, 1, d1, 0, 0, 0))
            {
                print("Saving avg Tensor file \n");

            }
        if (dimension==2)
            if (tfa.openWrite("avgTensor.tensor", mtype, 2, d1, d2, 0, 0))
            {
                print("Saving avg Tensor file \n");

            }
        if (dimension==3)
            if (tfa.openWrite("avgTensor.tensor", mtype, 3, d1,d2, d3, 0))
            {
                print("Saving avg Tensor file \n");

            }
        if (dimension==4)
            if (tfa.openWrite("avgTensor.tensor", mtype, 4, d1, d2, d3, d4))
            {
                print("Saving avg Tensor file \n");

            }
            print("m_n_samples %d\n",m_n_samples);
         //    Tprint(avgTensor);
        THFloatTensor_mul(avgTensor->t, 1.0/m_n_samples);
        Tensor *otensor=NULL;
        switch (mtype)
        {
        case Tensor::Short:
            if (dimension==1) otensor = new ShortTensor(d1);
            else if (dimension==2) otensor = new ShortTensor(d1,d2);
            else if (dimension==3) otensor = new ShortTensor(d1,d2,d3);
            else if (dimension==4) otensor = new ShortTensor(d1,d2,d3,d4);
            break;

        case Tensor::Int:
            if (dimension==1) otensor = new IntTensor(d1);
            else if (dimension==2) otensor = new IntTensor(d1,d2);
            else if (dimension==3) otensor = new IntTensor(d1,d2,d3);
            else if (dimension==4) otensor = new IntTensor(d1,d2,d3,d4);
            break;
        default:
            print("no type\n");

        }

        otensor->copy(avgTensor);
      //  Tprint(avgTensor);
        tfa.save(*otensor);
        tfa.close();

    }


    tf.close();
    return 0;
}
