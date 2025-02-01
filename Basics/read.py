import cv2 as cv
print(cv.__version__)

#loading an image from a file
# image = cv.imread('./Resources/Photos/cat_large.jpg')

# #Display image on window
# cv.imshow('Image', image)
# cv.waitKey(0)

# Reading Videos
capture = cv.VideoCapture('./Resources/Videos/dog.mp4')

while True:
    isTrue, frame = capture.read() #Capture the video frame by frame
    cv.imshow('Video', frame) #Display the video frame by frame

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()

cv.destroyAllWindows()