import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

def extractImages(folderpath, size = (128,128)):
    imgStack = []
    for filename in os.listdir(folderpath):
        img = cv2.imread(os.path.join(folderpath,filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size)
        imgStack.append(img)

    return imgStack


def genLabels(train1, train2):
    trainingData = np.concatenate((train1, train2), axis=0)
    trainingLabels = np.concatenate((np.zeros(len(train1), dtype=int), np.ones(len(train2), dtype=int)))

    indices = np.arange(len(trainingLabels))
    np.random.shuffle(indices)

    trainingData = trainingData[indices]
    trainingLabels = trainingLabels[indices]

    return trainingData, trainingLabels

acanthocyteTest = extractImages(r"C:\Users\dalki\PycharmProjects\Pathology\Acanthocytosis Supervised Learning\OutputRBCs\acanthocyteTest")
acanthocyteTrain = extractImages(r"C:\Users\dalki\PycharmProjects\Pathology\Acanthocytosis Supervised Learning\OutputRBCs\acanthocyteTrain")

notAcanTest = extractImages(r"C:\Users\dalki\PycharmProjects\Pathology\Acanthocytosis Supervised Learning\OutputRBCs\not_acanthocyteTest")
notAcanTrain = extractImages(r"C:\Users\dalki\PycharmProjects\Pathology\Acanthocytosis Supervised Learning\OutputRBCs\non_acanthocyteTrain")

trainImages, trainLabels = genLabels(acanthocyteTrain, notAcanTrain)
testImages, testLabels = genLabels(acanthocyteTest, notAcanTest)

trainImages = trainImages / 255.0
testImages = testImages / 255.0

classNames = ["Acanthocyte", "Not Acanthocyte"]

plt.figure(figsize=(10,10))
for i in range(64):
    plt.subplot(8,8,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(trainImages[i], cmap=plt.cm.binary)
    plt.xlabel(classNames[trainLabels[i]])
plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Reshape((128, 128, 1), input_shape=(128, 128)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(trainImages, trainLabels, epochs=20)

test_loss, test_acc = model.evaluate(testImages,  testLabels, verbose=2)

print('\nTest accuracy:', test_acc)



probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(testImages)


np.save('predictions.npy', predictions)
np.save('test_labels.npy', testLabels)
np.save('test_images.npy', testImages)

