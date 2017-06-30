import QtQuick 2.2
import QtQuick.Window 2.2
import "qrc:/qml"

Window { 
    id:mainWindow
    visible: true
    visibility: Window.Maximized

    Rectangle {
        anchors.fill: parent
        color:"grey"
        Rectangle{
            id: weightDisplay
            border.width: 2
            border.color: "grey"
            radius: 5
            width: parent.width*3/4
            height: parent.height/2
            anchors{
                top:parent.top
                left:parent.left
            }
            
        }

        Rectangle{
            id: lossDisplay
            border.width: 2
            border.color: "grey"
            radius: 5
            width: parent.width*3/4
            height: parent.height/2
            anchors{
                top:weightDisplay.bottom
                left:parent.left
            }
            
        }

        Rectangle{
            id:configPanel
            border.width: 2
            border.color: "grey"
            radius: 5
            width: parent.width/4
            height: parent.height
            anchors{
                top: parent.top
                left: weightDisplay.right
            }

            
            
        }

    }
    
}