import QtQuick 2.2
import QtQuick.Controls 1.2
import "qrc:/qml"

TabView {
    id: tabview
    width: parent.width
    height: parent.height

    property var tabCount:0

    Component.onCompleted:{
        applicationData.networkUpdate.connect(updateNetwork)
    } 

   function updateNetwork(){
        for( var i=0; i<tabCount; i++)
            removeTab(i)
        tabview.tabCount=0
        var names = applicationData.getBlobName()
        var component = Qt.createComponent("Layer.qml");
        for(var i in names ){
            console.log(names[i])
            var c_tab=currentIndex
            var t = addTab(names[i],component)
            currentIndex=i
            t.item.layerName = names[i]
            tabview.tabCount = tabview.tabCount+1
            currentIndex=c_tab
        }
    }
}