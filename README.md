# Real-time-facial-gender-recognition
link to the trained model: https://www.dropbox.com/s/d1yvmcpqfrwklsb/gender_recog.zip?dl=0

The model is trained via Transfer Learning, using InceptionV-3 model on around 145 Female faces and 200 Male faces.
Final test accuracy achieved: 78 %

# making it real time
haarcascade_frontalface_default.xml this file helps in finding face from the whole image and extract the ROI, then the ROI is 
reshaped to 112,92 (weidth, height).
Model is loaded in the memory

saving a frame to the disk every 10 frames, reading it and passing through the classifier
