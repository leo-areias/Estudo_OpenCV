import cv2 as cv

# img = cv.imread('./Resources/Photos/cat_large.jpg')
# cv.imshow('Cat', img)

# Funçao que muda a escala do Frame para 75% do original
def rescaleFrame(frame, scale=0.75):
    # Para videos, fotos e live videos
    width = int(frame.shape[1] * scale) # .shape[1] para width
    height = int(frame.shape[0] * scale) # .shape[0] para height
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)

def changeRes(width, height):
    # Para live video
    capture.set(3, width)
    capture.set(4, height)


capture = cv.VideoCapture('./Resources/Videos/dog.mp4')

while True:
    isTrue, frame = capture.read() #Capture the video frame by frame
    cv.imshow('Video', frame) #Display the video frame by frame

    frame_resized = rescaleFrame(frame, scale=0.2)
    cv.imshow('Video Resized', frame_resized)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()

cv.destroyAllWindows()

