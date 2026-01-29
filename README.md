6.1 Computer Vision 
The implementation phase involves setting up the system environment, running the 
Flask application, and integrating all modules responsible for vehicle detection, 
violation tracking, and license plate recognition. Once the Flask server is started, it 
initializes the trained models, loads the video stream, and begins real-time 
processing of traffic footage. The server console confirms that the application is 
successfully running and ready to perform detection operations.
<img width="898" height="154" alt="image" src="https://github.com/user-attachments/assets/c32c3231-9630-49bb-b685-66b83cdbbfee" />

The Traffic Signal Violation Detection System uses OpenCV and YOLOv8 for 
real-time computer vision processing.
<img width="879" height="450" alt="image" src="https://github.com/user-attachments/assets/031d8b03-4b8e-4fcc-a6c3-534eeae095b9" />

Page | 10    
Page | 11    
   
 
Figure 6.1: Flask-based Traffic Violation Detection Web Interface 
   
OpenCV (Open Source Computer Vision Library) is used to handle image 
and video processing operations such as reading video frames, resizing, 
drawing lines and bounding boxes, encoding frames for streaming, and basic 
image preprocessing for license plate recognition. 
 
The YOLOv8 model (developed by Ultralytics) is used for vehicle detection. 
It identifies and classifies vehicles such as cars, buses, trucks, and motorbikes 
in each frame. The SORT (Simple Online and Realtime Tracking) 
algorithm is then applied to assign unique tracking IDs to each vehicle, 
allowing the system to monitor them across multiple frames. 
Figure: Red-Light Violation Detected by the System 
A second YOLO model, specifically trained for license plate detection, is used 
to extract the region of interest (ROI) containing the vehicle’s plate when a 
violation occurs. The cropped plate image is then processed using OpenCV 
image preprocessing techniques (grayscale conversion and thresholding) 
before 
text 
recognition 
is 
performed by the OCR module 
(read_license_plate() function). 
The entire process runs within a Flask web application, which serves as the 
graphical user interface (GUI) for monitoring traffic in real time. The Flask 
server continuously streams processed video frames to the web browser, 
allowing the administrator to observe detected vehicles, identify violations, and 
monitor recognized license plate numbers dynamically. Additionally, the 
interface provides manual control options for switching the traffic signal 
between red and green, enabling flexible testing and demonstration of the 
system’s capabilities. 
Page | 12    
Figure: License Plate Detection and Text Recognition 
Figure 6.5: Stored Violation Data
