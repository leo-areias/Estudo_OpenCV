import os
import caer
import canaro
import numpy as np
import cv2 as cv
import gc
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers.legacy import SGD

# DATASET
# https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset/data
#Todas imagens tem que ser do mesmo tamanho para coloca-las no modelo

IMG_SIZE = (80,80)
channels = 1
char_path = "./archive/simpsons_dataset"

char_dict = {}
for char in os.listdir(char_path): # Pega cada arquivo que há na pasta principal
    char_dict[char] = len(os.listdir(os.path.join(char_path,char))) # Cria o dicionário com o arquivo e a quantidade de imagens que há nele

# Sort em ordem descendente
char_dict = caer.sort_dict(char_dict, descending = True)

characters = []
count = 0
for i in char_dict:
    characters.append(i[0])
    count += 1
    if count >= 10:
        break
print(characters)


# Dados de treino
train = caer.preprocess_from_dir(char_path, characters, channels = channels, IMG_SIZE = IMG_SIZE, isShuffle = True)


# import matplotlib.pyplot as plt
# plt.figure(figsize = (30,30))
# plt.imshow(train[0][0], cmap = 'gray')
# plt.show()

featureSet, labels = caer.sep_train(train, IMG_SIZE = IMG_SIZE)

# Normalizando o featureSet (range de (0,1))
featureSet = caer.normalize(featureSet)
labels = to_categorical(labels, len(characters))

x_train, x_val, y_train, y_val = caer.train_val_split(featureSet, labels, val_ratio=.2)

del train
del featureSet
del labels
gc.collect()

BATCH_SIZE = 32
EPOCHS = 10

# Gerador de imagens para introduzir novas variáveis
datagen = canaro.generators.imageDataGenerator()
train_gen = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

# Criando o modelo
output_dim=10

w, h = IMG_SIZE[:2]

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(w, h,channels)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu')) 
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))

# Output Layer
model.add(Dense(output_dim, activation='softmax'))

model.summary()

# Treinando o modelo
model.compile(loss='binary_crossentropy', metrics=['accuracy'])
callbacks_list = [LearningRateScheduler(canaro.lr_schedule)]
training = model.fit(train_gen,
                    steps_per_epoch=len(x_train)//BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(x_val,y_val),
                    validation_steps=len(y_val)//BATCH_SIZE,
                    callbacks = callbacks_list)

print(characters)


test_path = './archive/kaggle_simpson_testset/kaggle_simpson_testset/bart_simpson_40.jpg'
img = cv.imread(test_path)

def prepare(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, IMG_SIZE)
    img = caer.reshape(img, IMG_SIZE, 1)
    return img

predictions = model.predict(prepare(img))

print(characters[np.argmax(predictions[0])])

