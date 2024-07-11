import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from network.backbone import network_backbone, opt_loss
from keras.layers import Input
from Utilities.dataset import load_dataset, dataset_augmentation
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             roc_curve, auc)

# Dataset directory
dataset_dir = 'Datasets/vocal_non-vocal/'
# Load dataset
features_X, features_Y = load_dataset(dataset_dir)
# Reshape the label feature
features_Y = features_Y.reshape(features_Y.shape[0], 1)
# Split dataset into Training and Test set.
X_train, X_test, y_train, y_test = train_test_split(features_X, features_Y, test_size=0.2, stratify=features_Y, shuffle=True, random_state=42)

# This block is used when dataset is trained with K-fold cross-validation technique
inputs = np.concatenate((X_train, X_test), axis=0)
targets = np.concatenate((y_train, y_test), axis=0)
# Scale dataset into smaller value
minmax_scaler = preprocessing.MinMaxScaler()


# Declare some arrays and dictionaries to store important values
score_list = []
histories = {}
val_acc_hist, test_acc_hist = [], []

roc_hist = {}
fpr_list = []
tpr_list = []
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

# Train with 5-fold cross-validation
class_labels = ['non-vocalization', 'vocalization']
i = 1
k_fold = KFold(n_splits=5, shuffle=False, random_state=None)
for fold_idx, (train_index, val_index) in enumerate(k_fold.split(np.arange(len(inputs)))):
    fold_idx += 1
    print(f'Fold {fold_idx}')
    print('=========')
    X_input, X_val = inputs[train_index], inputs[val_index]
    y_input, y_val = targets[train_index], targets[val_index]
    print(len(X_input), len(y_input), len(y_val))

    # Augment only training set
    X_input, y_input = dataset_augmentation(X_input, y_input, feature_type=2, aug=True)
    X_val, y_val = dataset_augmentation(X_val, y_val, feature_type=2, aug=False)
    print(X_input.shape)
    print(y_val.shape)

    k_fold_model = network_backbone(Input(shape=X_input[0].shape, name="input1"))

    opt, loss_func = opt_loss()

    k_fold_model.compile(
        loss=loss_func,
        optimizer=opt,
        metrics=['accuracy']
    )

    history = k_fold_model.fit(
        X_input,
        y_input,
        epochs=50,
        batch_size=16,
        verbose=1,
        shuffle=True
    )

    histories[fold_idx] = history

    # Predict on validation set
    pred = k_fold_model.predict(X_val, batch_size=16)
    val_pred = np.argmax(pred, axis=1)
    y_val_true = np.argmax(y_val, axis=1)

    # Validation ROC curve block
    fpr, tpr, thresholds = roc_curve(y_val_true, val_pred)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='Fold %d (AUC = %0.4f)' % (i, roc_auc))
    i += 1

    roc_hist[fold_idx] = {'fpr': fpr_list, 'tpr': tpr_list}

    target_names = ['0', '1']
    val_acc = np.round(accuracy_score(y_val_true, val_pred), 4)
    val_report = classification_report(y_val_true, val_pred, target_names=class_labels, digits=4)
    val_acc_hist.append(val_acc)
    print(f'Validation accuracy for fold-{fold_idx} is {val_acc}')
    print('Validation Classification Reports')
    print(val_report)

print('5 K-fold validation accuracies:', val_acc_hist)

plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='black')

mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)

plt.plot(mean_fpr, mean_tpr, color='blue', label=r'Mean ROC (AUC = %0.4f)' % mean_auc, lw=1, alpha=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.text(0.32, 0.7, 'More accurate area', fontsize=12)
plt.text(0.63, 0.4, 'Less accurate area', fontsize=12)
plt.show()
