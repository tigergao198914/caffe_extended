#ifndef __NETWORK_WRAPPTER_H__
#define __NETWORK_WRAPPTER_H__

#include <QObject>
#include <iostream>
#include <qqmlengine.h>
#include <QQuickImageProvider>

#include <memory>
#include <string>
#include <iostream>

#include <caffe/caffe.hpp>
using namespace caffe; 

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class NetworkWrapper : public QObject, public QQuickImageProvider
{
    Q_OBJECT
public:

    NetworkWrapper();

    //QQuickImageProvider
    QPixmap requestPixmap(const QString &id, QSize *size, const QSize &requestedSize);

    Q_INVOKABLE void loadNetwork();

    Q_INVOKABLE QList<QString> getLayerName();

    Q_INVOKABLE QList<QString> getBlobName();

    Q_INVOKABLE void setInputImage(QString imageUrl);

    Q_INVOKABLE void setProtoFile(QString proto);

    Q_INVOKABLE void setWeightFile(QString weight);

    Q_INVOKABLE QStringList getLayerModel(QString layerName);       
private:
    std::string _inputImageUrl;
    std::string _prototxtFile;
    std::string _weightFile;
    std::shared_ptr< Net<float> > _net;

signals:
	void inputUpdate(QString);
    void networkUpdate();
};

#endif