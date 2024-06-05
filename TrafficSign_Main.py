import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import random
from keras.src.legacy.preprocessing.image import ImageDataGenerator

################# Parametreler #####################
path = "myList/myData" #sinif klasorlerini icerir.
labelFile = 'myList/labels.csv' #etiketler
batch_size_val=50  #toplu işleme için kullanılacak veri örneklerinin sayısı
steps_per_epoch_val=2000 #eğitim sırasında veri setinin kaç parçaya bölüneceğini belirler.
epochs_val=10 #10kez egitilir
imageDimesions = (32,32,3)
testRatio = 0.2    # %20test
validationRatio = 0.2 # kalangoruntunun %20 dogrulama
###################################################
############################### Görüntüyü Aktarma
count = 0
images = []
classNo = []
myList = os.listdir(path)
print("Total Classes Detected:", len(myList))
noOfClasses = len(myList)
print("Importing Classes.....")
for x in range(0, len(myList)):
    myPicList = os.listdir(path + "/" + str(count))
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(count) + "/" + y)
        images.append(curImg)
        classNo.append(count)
    print(count, end=" ")
    count += 1
print(" ")
images = np.array(images)
classNo = np.array(classNo)

############################### Verileri Bolme
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)

# X_train = EĞİTİM YAPILACAK GÖRÜNTÜ DİZİSİ
# y_train = İLGİLİ SINIF KİMLİĞİ

############################### HER VERİ SETİ İÇİN GÖRÜNTÜ SAYISINI ETİKET SAYISIYLA EŞLEŞTİRİP EŞLENDİĞİNİ KONTROL ETMEK İÇİN
print("Data Shapes")
print("Train", end="");
print(X_train.shape, y_train.shape)
print("Validation", end="");
print(X_validation.shape, y_validation.shape)
print("Test", end="");
print(X_test.shape, y_test.shape)
assert (X_train.shape[0] == y_train.shape[
    0]), "Eğitim setindeki görsellerin sayısı etiket sayısına eşit değil"
assert (X_validation.shape[0] == y_validation.shape[
    0]), "Görüntülerin sayısı doğrulama kümesindeki etiketlerin sayısına eşit değil"
assert (X_test.shape[0] == y_test.shape[0]), "Görüntülerin sayısı test setindeki etiketlerin sayısına eşit değil"
assert (X_train.shape[1:] == (imageDimesions)), " Eğitim görsellerinin boyutları yanlış "
assert (X_validation.shape[1:] == (imageDimesions)), " Doğrulama görüntülerinin boyutları yanlış "
assert (X_test.shape[1:] == (imageDimesions)), " Test görsellerinin boyutları yanlış"

############################### CSV DOSYASINI OKU
data = pd.read_csv(labelFile)
print("data shape ", data.shape, type(data))

############################### DISPLAY SOME SAMPLES IMAGES  OF ALL THE CLASSES
num_of_samples = []
cols = 5
num_classes = noOfClasses
fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 300))
fig.tight_layout()
for i in range(cols):
    for j, row in data.iterrows():
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j) + "-" + row["Name"])
            num_of_samples.append(len(x_selected))

############################### Grafik
print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Eğitim veri setinin dağıtımı")
plt.xlabel("Sınıf No")
plt.ylabel("Resim sayısı")
plt.show()


############################### GÖRÜNTÜLERİN ÖN İŞLENMESİ

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)  # GRİ TONLAMAYA DÖNÜŞTÜR
    img = equalize(img)  # BİR GÖRÜNTÜDE AYDINLATMAYI STANDARTLAŞTIR
    img = img / 255  # normalizasyon
    return img


X_train = np.array(list(map(preprocessing, X_train)))
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))
cv2.imshow("GrayScale Images",
           X_train[random.randint(0, len(X_train) - 1)])  # EĞİTİMİN DOĞRU YAPILDIĞINI KONTROL ETMEK İÇİN

############################### 1derinlik ekle
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

############################### GÖRÜNTÜLERİN BÜYÜTÜLMESİ
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)
batches = dataGen.flow(X_train, y_train,
                       batch_size=20)
X_batch, y_batch = next(batches)

# GENİŞLETİLMİŞ GÖRÜNTÜ ÖRNEKLERİNİ GÖSTERMEK İÇİN
fig, axs = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()

for i in range(15):
    axs[i].imshow(X_batch[i].reshape(imageDimesions[0], imageDimesions[1]))
    axs[i].axis('off')
plt.show()

y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)


############################### CONVOLUTION NEURAL NETWORK MODEL
def myModel():
    no_Of_Filters = 60
    size_of_Filter = (5, 5)
    size_of_Filter2 = (3, 3)
    size_of_pool = (2, 2)
    no_Of_Nodes = 500
    model = Sequential()
    model.add((Conv2D(no_Of_Filters, size_of_Filter, input_shape=(imageDimesions[0], imageDimesions[1], 1),
                      activation='relu')))
    model.add((Conv2D(no_Of_Filters, size_of_Filter, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))

    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(no_Of_Nodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


############################### egitim
model = myModel()
print(model.summary())
history = model.fit(dataGen.flow(X_train, y_train, batch_size=batch_size_val),
                              steps_per_epoch=steps_per_epoch_val, epochs=epochs_val,
                              validation_data=(X_validation, y_validation), shuffle=1)

###############################
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])


model.save("model_trained.h5")
cv2.waitKey(0)