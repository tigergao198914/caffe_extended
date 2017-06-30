#include "network_wrapper.h"

NetworkWrapper::NetworkWrapper()
    : QQuickImageProvider(QQuickImageProvider::Pixmap)
{
    _prototxtFile = "/home/hosery/git/TrackVideo/pretrained/GOTURN/tracker.prototxt";
    _weightFile   = "/home/hosery/git/TrackVideo/pretrained/GOTURN/tracker.caffemodel";
}

QPixmap NetworkWrapper::requestPixmap(const QString &id, QSize *size, const QSize &requestedSize)
{
    int width = 100;
    int height = 50;
    std::string stdId = id.toUtf8().constData();
    std::cout<< "requestPixmap:"<<stdId<<std::endl;
    QColor color(id);
    if( color.isValid() )
    {
        QPixmap pixmap(requestedSize.width() > 0 ? requestedSize.width() : width,
                        requestedSize.height() > 0 ? requestedSize.height() : height);
        pixmap.fill(QColor(id).rgba());
        return pixmap;
    }
    else if( stdId.find("weight") == 0 )
    {
        if(stdId.find("default")!=-1)
        {
           QPixmap pixmap(requestedSize.width() > 0 ? requestedSize.width() : width,
                        requestedSize.height() > 0 ? requestedSize.height() : height);
            pixmap.fill(QColor("black").rgba());
            return pixmap;
        }

        // weight/layerName/featureIndex
        int idLength = stdId.length();
        int slashTime = 0;
        int layerNameStartPos = 0;
        int featureIndexStartPos = 0;
        for( int i=0; i<idLength; i++ )
        {
            if( stdId[i] == '/' ){
                slashTime++;
                if( slashTime==1 ){
                    layerNameStartPos = i+1;
                }
                else if( slashTime==2 )
                {
                    featureIndexStartPos = i+1;
                }
            }
        }

        std::string layerName = stdId.substr( layerNameStartPos, featureIndexStartPos-layerNameStartPos-1 );
        std::string featureIndex = stdId.substr( featureIndexStartPos, idLength-featureIndexStartPos ); 

        std::cout<< layerName <<":" << featureIndex << std::endl;

        int layerIndex = std::stoi(layerName);
        int index = std::stoi(featureIndex);

        Blob<float> *layer = _net->learnable_params()[layerIndex];
        int target_width = layer->width();
        int target_height = layer->height();
        float* target_data = layer->mutable_cpu_data();

        int i = 0;
        switch( layerIndex )
        {
            case 0:
                i++;
                break;
            case 1:
                i++;
                break;
            case 2:
                i++;
                break;
            case 3:
                i++;
                break;
            case 4:
                i++;
                break;
            case 5:
                i++;
                break;
            case 6:
                i++;
                break;
            case 7:
                i++;
                break;
            case 8:
                i++;
                break;
            case 9:
                i++;
                break;
        }

        target_data += index * target_width * target_height;
        cv::Mat curImage(target_height, target_width, CV_32FC1, target_data, target_width);

        double min, max;
        cv::minMaxLoc(curImage, &min, &max);
        curImage.convertTo(curImage,CV_8U,255.0/(max-min),-255.0*min/(max-min));

        //curImage.convertTo(curImage, CV_8UC1);
        QImage image( (unsigned char *)curImage.data, target_width, target_height,QImage::Format_Grayscale8);
        return QPixmap::fromImage(image);      
    }
    else if( stdId.find("active") == 0 )
    {
        // active/layerName/featureIndex
        std::string layerName;
        std::string featureIndex;
    }

    QPixmap pixmap(id);
    return pixmap;
}



void NetworkWrapper::loadNetwork()
{
    std::cout<< "load Network ..."<<std::endl;
    std::cout<<"prototxt:"<<_prototxtFile<<std::endl;
    std::cout<<"weight:"<<_weightFile<<std::endl;
    
    /* Load the network. */
    _net.reset(new Net<float>(_prototxtFile, TEST));
    _net->CopyTrainedLayersFrom(_weightFile);
    emit networkUpdate();
}

QList<QString> NetworkWrapper::getBlobName()
{
    QList<QString> list;
    if( _net == nullptr )
    {
        return list;
    }
    #if 0
    std::vector<std::string> blobs = _net->blob_names();
    for( auto iter=blobs.begin(); iter!=blobs.end(); iter++ )
    {
        list << (*iter).c_str();
    }
    #else
    vector<Blob<float>*> v = _net->learnable_params();
    for( int i=0; i<v.size();i++ )
    {
        if( _net->learnable_params()[i]->num_axes()!=4 )
        {
            continue;
        }
        list << std::to_string(i).c_str();
    }
    #endif
    return list;
}

QList<QString> NetworkWrapper::getLayerName()
{
    QList<QString> list;
    if( _net == nullptr )
    {
        return list;
    }
    std::vector<std::string> layers = _net->layer_names();
    for( auto iter=layers.begin(); iter!=layers.end(); iter++ )
    {
        list << (*iter).c_str();
    }
    return list;
}

void NetworkWrapper::setInputImage( QString imageUrl )
{
    std::string tmp = imageUrl.toUtf8().constData();
    _inputImageUrl = tmp.substr(6,tmp.size());
    std::cout<<"set input image:"<<_inputImageUrl<<std::endl;
    emit inputUpdate(QString(_inputImageUrl.c_str()));
}

void NetworkWrapper::setProtoFile(QString proto)
{
    std::string tmp = proto.toUtf8().constData();
    _prototxtFile = tmp.substr(7,tmp.size());
    std::cout<< "proto file:"<< _prototxtFile<< std::endl;
}

void NetworkWrapper::setWeightFile(QString weight)
{
    std::string tmp = weight.toUtf8().constData();
    _weightFile = tmp.substr(7,tmp.size());
    std::cout<< "weight file:"<< _weightFile<< std::endl;
}

//inline const vector<Blob<Dtype>*>& learnable_params()
QStringList NetworkWrapper::getLayerModel(QString layerName)
{
    std::string tmp = layerName.toUtf8().constData();
    std::cout<<"get layer model:"<<tmp<<std::endl;
    QStringList s;
    if( tmp.compare("default")==0 )
    {
        return s;
    }

    int index = std::stoi(tmp);
    vector<Blob<float>*> v = _net->learnable_params();
    Blob<float> *param = v[index];
    

    if(param==nullptr || param->num_axes()!=4 )
        return s;

    int channelNum = param->shape(0)*param->shape(1);
    std::cout<<"get layer channelNum:"<<channelNum<<std::endl;
    for(int i=0; i<channelNum; i++)
    {
        s<< std::to_string(i).c_str();
    }
    return s;
}