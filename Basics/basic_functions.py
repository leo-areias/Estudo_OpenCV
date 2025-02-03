import cv2 as cv

img = cv.imread('./Resources/Photos/park.jpg')
cv.imshow('Park', img)

#Convert to grey scale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Bluring image
blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

#Edge cascade (ajuda na detecção de objetos, para detectar padrões nas iamgens)
canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny', canny)

# Dilating image
dilated = cv.dilate(canny, (7,7), iterations = 3)
cv.imshow('Dilated', dilated)

#Eroding
eroded = cv.erode(dilated, (3,3), iterations = 1)
cv.imshow('Eroded', eroded)

# Resize
resized = cv.resize(img, (500,500), interpolation = cv.INTER_CUBIC) #INTER_AREA best for resizing to lower frames
cv.imshow('Resized', resized)                                      #INTER_CUBIC and INTER_LINEAR best for scaling for large dimensions

#Cropping (You can select a portion of the image to slice)
cropped = img[50:200, 200:400]
cv.imshow('Cropped', cropped)

cv.waitKey(0)


