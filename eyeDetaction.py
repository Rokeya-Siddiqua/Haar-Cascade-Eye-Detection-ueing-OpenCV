import cv2

#face detactor
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#smile detactor
eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')


######### smile detect for a video start ###########
webcam = cv2.VideoCapture(0)  #video_name.mp4

while True:
    sucessful_frame_read, frame = webcam.read()
    #abort if there is an error
    if not sucessful_frame_read:
        break
    # frame in gray scale
    gray_scaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    #detect face and smile
    faces = face_detector.detectMultiScale(gray_scaled_frame)
    # run face detection within each of these faces
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),5)
        #create the face sub-image
        face = frame[y:y+h, x:x+w]
        #gray scale the face
        face_gray_scale = cv2.cvtColor(face, cv2.COLOR_BGR2BGRA)

        eyes = eye_detector.detectMultiScale(gray_scaled_frame, scaleFactor = 1.1, minNeighbors = 20)
        
        for (x_eye,y_eye,w_eye,h_eye) in eyes:
            #draw a rectangle around the eyes
            cv2.rectangle(face, (x_eye,y_eye),(x_eye + w_eye, y_eye + h_eye),(50,55,200),5)

    #display
    cv2.imshow('why so serious', frame)
    key = cv2.waitKey(1)

    #stop if Q/q key is pressed
    if key==81 or key==113:
        break

# clean up
webcam.release()
cv2.destroyAllWindows()  

    

        
 
######### smile detect for a video end ###########
print("code completed")