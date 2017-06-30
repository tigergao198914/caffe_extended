#include <iostream>
#include <QApplication>
#include <QQmlApplicationEngine>
#include <QWindow>
#include <qqmlengine.h>
#include <qqmlcontext.h>
#include <qqml.h>
#include <QtQuick/qquickitem.h>
#include <QtQuick/qquickview.h>
#include "network_wrapper.h"

using namespace std;

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    QQmlApplicationEngine engine(NULL);

    NetworkWrapper *network = new NetworkWrapper;
    engine.rootContext()->setContextProperty("applicationData", network);
    engine.addImageProvider("colors", network);

    engine.load(QUrl("qrc:qml/Main.qml"));

    return app.exec();
}