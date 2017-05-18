import QtQuick 2.2
import QtQuick.Controls 1.2

Rectangle{
    id: scrollview
    anchors.fill:parent
    property string layerName:"default"
    color: "green"
    GridView {
        id: gridview
        anchors.fill: parent
        cellWidth: 50; cellHeight: 50
        focus: true
        model: applicationData.getLayerModel(parent.layerName)

        //highlight: Rectangle { width: 80; height: 80; color: "lightsteelblue" }

        delegate: Item {
            Rectangle{
                width: gridview.cellWidth
                height: gridview.cellHeight
                border.color: "green"
                border.width: 3
                Image {
                    anchors.centerIn:parent
                    width: parent.width - parent.border.width
                    height: parent.height - parent.border.width
                    source:"image://colors/weight/"+layerName+"/"+index
                }
            }
            MouseArea {
                anchors.fill: parent
                onClicked: parent.GridView.view.currentIndex = index
            }
        }
    } 

    ScrollBar {
        id: listScrollBar
        orientation:  Qt.Vertical
        height: gridview.height
        width:  8
        scrollArea: gridview
        anchors.right: gridview.right
    } 
}