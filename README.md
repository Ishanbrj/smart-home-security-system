##  ESP32-CAM Streaming
After uploading the ESP32 code, open the Serial Monitor to get the camera IP address.

Example:  
`http://192.168.38.158/cam-hi.jpg`

##  Running Face Recognition
Run the following Python script to start the face recognition system:

```python
python face_recognition.py

url = 'http://192.168.38.158/cam-hi.jpg'
