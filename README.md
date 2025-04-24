# face-detection-comparison
This program compares the Viola-Jones and the MTCNN face detection algorithms. The program has 3 features:
1. Upload an image and process it with both algorithms. After processing, the application frames the detected faces with rectangles, and marks the most important features of the face with dots.
2. Turn on the webcam and track the faces in real time with both algorithms. Faces are marked in the same way as in the previous feature.
3. The program runs the algorithms one after the other on several images with different features (the images should be put in an _images_ folder in the root of the project), then records the number of detected faces, the number of false positives, and the time required for the run. The statistical evaluation of the results is shown with histograms.
