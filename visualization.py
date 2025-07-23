import numpy as np
import matplotlib.pyplot as plt

predictions = np.load('predictions.npy')
testLabels = np.load('test_labels.npy')
testImages = np.load('test_images.npy')
classNames = ["Acanthocyte", "Not an Acanthocyte"]


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    confidence = 100 * np.max(predictions_array)

    color = 'blue' if predicted_label == true_label else 'red'

    plt.xlabel(f"{classNames[predicted_label]} {confidence:2.0f}% ({classNames[true_label]})",
               color=color)


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(2))
    plt.yticks([])
    thisplot = plt.bar(range(2), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


num_images = len(testLabels)

# Calculate grid size dynamically (try to make it roughly square)
num_cols = 8
num_rows = (num_images + num_cols - 1) // num_cols  # ceiling division

plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], testLabels, testImages)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], testLabels)
plt.tight_layout()
plt.show()
