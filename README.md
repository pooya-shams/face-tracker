# Face Tracker

## Author: Pooya Shams kolahi

## Brief

This project is called face tracker, the program
basically tracks your face and according to the
difference of the position of nose tip and the
center of the frame captured by the camera, it
tries to understand if you are looking down or up
and then moves the text that you are reading or
editing, correspondingly.

## License

Licensed under MIT License. you can freely use, modify, contribute and distribute this project but let me know if you are doing sth interesting.

## Usage

the face tracker tracks your face and according to the difference of your nose tip and the center of the frame captured by camera decides if you want to move text up or down.  
you can control it's behaviour using four keyboard shortcuts(for now just four):
1.control+command+e/ctrl+super+e/ctrl+win+e: will show/hide the screen showing your face.  
2.control+command+v/ctrl+super+v/ctrl+win+v: will toggle vscode mode. (vscode mode tells the program if it should use vscode mode for scrolling (ctrl+\[up/down\]) or it should use normall mouse scroll mode)  
3.control+command+a/ctrl+super+a/ctrl+win+a: will toggle accurancy mode. (more accurate mode uses the 5 point facial landmark dataset and it uses more recources to run but less accurate mode just uses a face rect and is faster.)  
4.control+command+s/ctrl+super+s/ctrl+win+s: will start/stop the program.  
you can quit the program by using the \[control+command+q/ctrl+super+q/ctrl+win+q:\] keyboard shortcut. (note that if you have enabled the visual mode, you have to quit the window showing your face by pressing q in that window)  

## Requirements

python >= 3.6  
imutils >= 0.5.3  
pynput >= 1.6.8  
numpy >= 1.17.4  
dlib >= 19.18.0  
opencv (cv2) >= 4.1.2.30  
