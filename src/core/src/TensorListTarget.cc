#include "core/TensorListTarget.h"

namespace Torch
{

    TensorListTarget::TensorListTarget()
    {
        n_files = 0;
        n_examples=0;
        m_data=NULL; //new MemoryDataSet();
        dimension=0;
        d1=0;
        d2=0;
        d3=0;
        d4=0;

    }

    bool TensorListTarget::process(FileListCmdOption *tensorList_files, ShortTensor *target,Tensor::Type mtype, char *targetfile)
    {



        n_files = tensorList_files->n_files;
        TensorFile tf1;
        for (int i=0;i<n_files;i++)
        {

            if (tf1.openRead(tensorList_files->file_names[i])==false)
            {
                print("Error while reading torch file %s\n",tensorList_files->file_names[i]);
                return false;
            }
            const TensorFile::Header& header1 = tf1.getHeader();
            n_examples += header1.m_n_samples;
            m_type =  header1.m_type;
            dimension = header1.m_n_dimensions;
            d1 = header1.m_size[0];
            d2 = header1.m_size[1];
            d3 = header1.m_size[2];
            d4 = header1.m_size[3];
            tf1.close();

        }
        if (m_data !=NULL)
            delete m_data;

        m_data = new MemoryDataSet(n_examples, mtype, true, Tensor::Short);


        switch (m_type)
        {
        case 0:
            switch (dimension)
            {
            case 1:
                tensor = new CharTensor(d1);
                break;
            case 2:
                tensor = new CharTensor(d1,d2);
                break;
            case 3:
                tensor = new CharTensor(d1,d2,d3);
                break;
            case 4:
                tensor = new CharTensor(d1,d2,d3,d4);
                break;
            default:
            print("TensorListTarget::process() problem\n");

            }
            break;

        case 1:
            switch (dimension)
            {
            case 1:
                tensor = new ShortTensor(d1);
                break;
            case 2:
                tensor = new ShortTensor(d1,d2);
                break;
            case 3:
                tensor = new ShortTensor(d1,d2,d3);
                break;
            case 4:
                tensor = new ShortTensor(d1,d2,d3,d4);
                break;
                default:
                print("TensorListTarget::process() problem\n");
            }
            break;


        case 2:
            switch (dimension)
            {
            case 1:
                tensor = new IntTensor(d1);
                break;
            case 2:
                tensor = new IntTensor(d1,d2);
                break;
            case 3:
                tensor = new IntTensor(d1,d2,d3);
                break;
            case 4:
                tensor = new IntTensor(d1,d2,d3,d4);
                break;
                default:
                print("TensorListTarget::process() problem\n");
            }
            break;


        case 3:
            switch (dimension)
            {
            case 1:
                tensor = new LongTensor(d1);
                break;
            case 2:
                tensor = new LongTensor(d1,d2);
                break;
            case 3:
                tensor = new LongTensor(d1,d2,d3);
                break;
            case 4:
                tensor = new LongTensor(d1,d2,d3,d4);
                break;
                default:
                print("TensorListTarget::process() problem\n");
            }
            break;

        case 4:
            switch (dimension)
            {
            case 1:
                tensor = new FloatTensor(d1);
                break;
            case 2:
                tensor = new FloatTensor(d1,d2);
                break;
            case 3:
                tensor = new FloatTensor(d1,d2,d3);
                break;
            case 4:
                tensor = new FloatTensor(d1,d2,d3,d4);
                break;
                default:
                print("TensorListTarget::process() problem\n");
            }
            break;


        case 5:
            switch (dimension)
            {
            case 1:
                tensor = new DoubleTensor(d1);
                break;
            case 2:
                tensor = new DoubleTensor(d1,d2);
                break;
            case 3:
                tensor = new DoubleTensor(d1,d2,d3);
                break;
            case 4:
                tensor = new DoubleTensor(d1,d2,d3,d4);
                break;
                default:
                print("TensorListTarget::process() problem\n");
            }
            break;
        default:
            print("TensorListTarget::process() problem\n");





        }

        //now can fill the m_data



        int nexmp;
        int nc=0; //keeping track of count
        File tarfile;
        tarfile.open(targetfile,"r");
        for (int i=0;i<n_files;i++)
        {
        	int tarval;
        	tarfile.scanf("%d",&tarval);
        	//(*target)(0) = tarval;
        	ShortTensor *ta1 = new ShortTensor(1);
        	ta1->fill(tarval);
        	//target->fill(tarval);
        	print("Target Value %d\n",tarval);

            if (tf1.openRead(tensorList_files->file_names[i])==false)
            {
                print("TensorListTarget::process() Error while reading torch file %s\n",tensorList_files->file_names[i]);
                return false;
            }
            const TensorFile::Header& header1 = tf1.getHeader();
            nexmp = header1.m_n_samples;
            if ( m_type !=  header1.m_type ||   dimension != header1.m_n_dimensions ) //||
                  //  d1 != header1.m_size[0] ||    d2 != header1.m_size[1] ||  d3 != header1.m_size[2] ||
                 //   d4 != header1.m_size[3])
            {
                print("TensorListTarget::process() Inconsistency in torch file %s\n",tensorList_files->file_names[i]);
                return false;
            }
            for (int j=0;j<nexmp;j++)
            {

                switch (dimension)
                {
                case 1:
                    m_data->getExample(nc)->resize(d1);
                    break;

                case 2:
                    m_data->getExample(nc)->resize(d1,d2);
                    break;
                case 3:
                    m_data->getExample(nc)->resize(d1,d2,d3);
                    break;

                case 4:
                    m_data->getExample(nc)->resize(d1,d2,d3,d4);
                    break;
                default :
                    print("TesnorList::process No dimension \n");


                }
		//Tensor *tensor1;
		tensor->resize(header1.m_size[0],header1.m_size[1]);
                tf1.load(*tensor);
                m_data->getExample(nc)->copy(tensor);

                m_data->setTarget(nc, ta1);
               // ShortTensor *ts = (ShortTensor*)m_data->getTarget(nc);
		//print("Target Value %d\n",(*ts)(0));
		//m_data->



                nc++;
            }
            tf1.close();

        }
        tarfile.close();
        return true;



    }

    DataSet *TensorListTarget::getOutput()
    {
//    	for(int i=0;i<m_data->getNoExamples();i++)
//    	{
//    		ShortTensor*ts = (ShortTensor*)m_data->getTarget(i);
//    		print("Target Value GOT %d\n",(*ts)(0));
//    	}
        return m_data;
    }

    TensorListTarget::~TensorListTarget()
    {
//n_files = 0;
        delete m_data;
        delete tensor;
    }

}
