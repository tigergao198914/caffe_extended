import QtQuick 2.2
import QtQuick.Controls 1.2
import QtQuick.Dialogs 1.1
import QtQuick.Layouts 1.1
import QtQuick.Window 2.2

Rectangle{
    id: selector
    width: parent.width
    height: 50


    FileDialog 
    {
        id: fileDialog
        title: "Please choose a file"
        folder: shortcuts.home
        property var fileType: 0
        onAccepted: {
            if(fileType==1){
                protocol.text = ""+fileDialog.fileUrl
                applicationData.setProtoFile(protocol.text)
            }
            else if(fileType==2 ){
                weight.text = ""+fileDialog.fileUrl
                applicationData.setWeightFile(weight.text)
            }
            else if(fileType==3 ){
                image.text = ""+fileDialog.fileUrl
                applicationData.setInputImage( image.text );
            }
        }
        onRejected: {
            console.log("Canceled")
        }
    }

    Rectangle{
        id: rectangle1
        anchors.left: parent.left
        width: parent.width/4
        height: parent.height
        anchors.verticalCenter: parent.verticalCenter
        Button {
            id: protocolSelector
            anchors.left : parent.left
            text: "Choose a prototxt file"
            anchors.verticalCenter: parent.verticalCenter
            onClicked: { 
                fileDialog.fileType=1; 
                fileDialog.open()
            }
        }
    
        Text{
            id: protocol
            anchors.left: protocolSelector.right
            anchors.verticalCenter: parent.verticalCenter
        }
    }


    Rectangle{
        id: rectangle2
        width: parent.width/4
        height: parent.height
        anchors.left: rectangle1.right
        anchors.verticalCenter: parent.verticalCenter
        Button {
            id: weightSelector
            anchors.left: parent.left
            text: "Choose a weight file"
            anchors.verticalCenter: parent.verticalCenter
            onClicked: { 
                fileDialog.fileType=2; 
                fileDialog.open()
            }
        }

        Text{
            id: weight
            anchors.left: weightSelector.right
            anchors.verticalCenter: parent.verticalCenter
        }
    }

     Rectangle{
        id: rectangle3
        width: parent.width/4
        height: parent.height
        anchors.left: rectangle2.right
        anchors.verticalCenter: parent.verticalCenter
        Button {
            id: imageSelector
            anchors.left: parent.left
            text: "Choose a input image"
            anchors.verticalCenter: parent.verticalCenter
            onClicked: { 
                fileDialog.fileType=3; 
                fileDialog.open()
            }
        }

        Text{
            id: image
            anchors.left: imageSelector.right
            anchors.verticalCenter: parent.verticalCenter
        }
    }


     Rectangle{
        id: rectangle4
        width: parent.width/4
        height: parent.height
        anchors.left: rectangle3.right
        anchors.verticalCenter: parent.verticalCenter
        Button {
            id: loadBtn
            anchors.left: parent.left
            text: "load network"
            anchors.verticalCenter: parent.verticalCenter
            onClicked: { 
                applicationData.loadNetwork();
            }
        }
    }
}