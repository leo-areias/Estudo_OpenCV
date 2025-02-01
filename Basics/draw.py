import cv2 as cv
import numpy as np

# Cria uma imagem preta
blank = np.zeros((500,500,3), dtype='uint8')
cv.imshow('Blank', blank)

# # Paint the image a colour
# blank[200:300, 300:400] = 0,255,0
# cv.imshow('Green', blank)

# Criando um retangulo
cv.rectangle(blank, (0,0), (250,250), (0,0,255), thickness = -1)
# thickness pode ser usado para preencher o espaco com -1 ou cv.FILLED
cv.imshow('Rectangle', blank)

# Criando um circulo
cv.circle(blank, (250,250), 40, (0,255,0), thickness = -1)
cv.imshow('Circle', blank)

# Linha
cv.line(blank, (200,300), (200,200), (255,255,255), thickness = 3)
cv.imshow('Line', blank)

# Escrevendo texto
cv.putText(blank, 'Aprendendo OpenCV', (90,400), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255,255,255), thickness = 2)
cv.imshow('Text', blank)

cv.waitKey(0)
