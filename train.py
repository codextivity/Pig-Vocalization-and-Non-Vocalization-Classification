import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint
from network.backbone import network_backbone, opt_loss
from keras.layers import Input
from Utilities.dataset import load_dataset, dataset_augmentation
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix,
                             ConfusionMatrixDisplay)

# Dataset directory
dataset_dir = 'Datasets/vocal_non-vocal/'
# Load dataset
features_X, features_Y = load_dataset(dataset_dir)
# Reshape the label feature
features_Y = features_Y.reshape(features_Y.shape[0], 1)
# Split dataset into Training and Test set.
X_train, X_test, y_train, y_test = train_test_split(features_X, features_Y, test_size=0.2, stratify=features_Y, shuffle=True, random_state=42)


class_labels = ['non-vocalization', 'vocalization']

train_X, train_Y = dataset_augmentation(X_train, y_train, feature_type=5, aug=True)
test_X, test_Y = dataset_augmentation(X_test, y_test, feature_type=5, aug=False)

# Save log into tensorboard
log_name = 'MelSpectrogram'
current_date_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = 'Logs/'+current_date_time+'-'+log_name
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

opt, loss_func = opt_loss()

base_model = ModelCheckpoint('Models/'+log_name+'.h5',
                             monitor='val_accuracy',
                             mode='max',
                             verbose=1,
                             save_best_only=True)

model = network_backbone(Input(shape=train_X[0].shape, name="input1"))
model.compile(
    loss=loss_func,
    optimizer=opt,
    metrics=['accuracy'])

history = model.fit(
    train_X,
    train_Y,
    epochs=50,
    batch_size=16,
    verbose=1,
    shuffle=True,
    validation_split=0.2,
    callbacks=[base_model, tensorboard_callback])

total_lost_score = model.evaluate(test_X, test_Y, verbose=0)
print('Total test loss', total_lost_score[0])
print('Total test accuracy', total_lost_score[1])


# Display loss and accuracy graph
metrics = history.history
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['train_loss', 'val_loss'])
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch')
plt.ylabel('Loss [CrossEntropy]')
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history.epoch, 100*np.array(metrics['accuracy']), 100*np.array(metrics['val_accuracy']))
plt.legend(['train_accuracy', 'val_accuracy'])
plt.ylim([0, 100])
plt.xlabel('Epoch')
plt.ylabel('Accuracy [%]')
plt.title('Accuracy')

plt.suptitle('Loss and Accuracy Graph')
plt.show()

# Display Confusion matrix table

predicts = model.predict(test_X, batch_size=16)
test_predicts = np.argmax(predicts, axis=1)
test_y_true = np.argmax(test_Y, axis=1)

test_accuracy = np.round(accuracy_score(test_y_true, test_predicts), 4)
test_report = classification_report(test_y_true, test_predicts, target_names=class_labels, digits=4)
print(test_report)

test_confusion_matrix = confusion_matrix(test_y_true, test_predicts)
# Normalize confusion matrix
norm_test_confusion_matrix = test_confusion_matrix.astype('float') / test_confusion_matrix.sum(axis=1)[:, np.newaxis]

# Display normal confusion matrix
dist_1 = ConfusionMatrixDisplay(confusion_matrix=test_confusion_matrix, display_labels=class_labels)
dist_1.plot()
dist_2 = ConfusionMatrixDisplay(confusion_matrix=norm_test_confusion_matrix, display_labels=class_labels)
plt.show()




