import QtQuick 2.2
import QtQuick.Window 2.2
import "qrc:/qml"

Window { 
    id:mainWindow
    visible: true
    visibility: Window.Maximized

    Component.onCompleted: {
        applicationData.inputUpdate.connect(updateInputImage) 
    }

   function updateInputImage(filename) {
       originImage.source = "image://colors"+filename
    }

    Protocol{
        id: protocol
        anchors.bottom: parent.bottom
    }

    Rectangle{
        width:parent.width
        height:parent.height-protocol.height
        Image{
            id: originImage
            anchors.top:parent.top
            anchors.left:parent.left
            width: parent.width/5
            height: parent.height/3
            source:"image://colors/black"
        }
        
        Image{
            id: selectChannel
            anchors.top:originImage.bottom
            width: parent.width/5
            height: parent.height/3
            source:"image://colors/white"
        }

        Image{
            id: diffImage
            anchors.top:selectChannel.bottom
            width: parent.width/5
            height: parent.height/3
            source:"image://colors/black"
        }

        Rectangle{
            id: layers
            anchors.left: originImage.right
            width:  parent.width*3/5.
            height: parent.height

            LayerTab{
            }
        }  

        Rectangle{
            id: gradientDesent
            width:parent.width/5
            height:parent.height/3
            anchors.left: layers.right
            anchors.top: parent.top
            color:"black"
        }  
    }
}