import sys
import pandas as pd
from time import time, ctime
from utils_imgproc import image_preprocessing
from utils_callback import eval_loss, callbacks_during_train
import shutil
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
import os
from utils_gen import gen_paths_img_dm, gen_var_from_paths
from utils_imgproc import norm_by_imagenet
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
net = 'countingnet'
dataset = 'A'
weights_dir = 'data/Abalone/weights_' + dataset
LOSS = 'MSE_BCE'

(test_img_paths, train_img_paths), (test_dm_paths, train_dm_paths) = gen_paths_img_dm(
    path_file_root='data/Abalone/',
    dataset=dataset
)
print("test_img_paths:", len(test_img_paths))
print("train_img_paths:", len(train_img_paths))
print("test_dm_paths:", len(test_dm_paths))
print("train_dm_paths:", len(train_dm_paths))
# Generate raw images(normalized by imagenet rgb) and density maps
test_x, test_y = gen_var_from_paths(test_img_paths[:]), gen_var_from_paths(test_dm_paths[:], stride=2)
print("test_x.shape", test_x.shape)
print("test_y.shape", test_y.shape)
test_x = norm_by_imagenet(test_x)  # Normalization on raw images in test set, those of training set are in image_preprocessing below.
print('Test data size:', test_x.shape[0], test_y.shape[0], len(test_img_paths))
train_x, train_y = gen_var_from_paths(train_img_paths[:]), gen_var_from_paths(train_dm_paths[:], stride=2)
print("train_x.shape", train_x.shape)
print("train_y.shape", train_y.shape)
print('Train data size:', train_x.shape[0], train_y.shape[0], len(train_img_paths))

# Analysis on results
dis_idx = 16 if dataset == 'B' else 0
weights_dir_neo = 'weights_A_MSE_BCE_bestMAE0.344_Wed-Oct-23'
model = model_from_json(open('data/Mussel/models/{}.json'.format(net), 'r').read())
#model.load_weights(os.path.join(weights_dir_neo, '{}_best.hdf5'.format(net)))
model.load_weights("/mnt/jackfrost/counting/weights_A_MSE_BCE_bestMAE0.144_Thu-Oct-24/countingnet_MAE124.285_RMSE201.224_SFN0.0_MAPE0.144_epoch267-4.0.hdf5")
ct_preds = []
ct_gts = []
for i in range(len(test_x[:])):
    if i % 100 == 0:
        print('{}/{}'.format(i, len(test_x)))
    i += 0
    test_x_display = np.squeeze(test_x[i])
    test_y_display = np.squeeze(test_y[i])
    path_test_display = test_img_paths[i]
    pred = np.squeeze(model.predict(np.expand_dims(test_x_display, axis=0)))
    ct_pred = np.sum(pred)
    ct_gt = np.sum(test_y_display)
    ct_preds.append(ct_pred)
    ct_gts.append(ct_gt)
plt.plot(ct_preds, 'r>')
plt.plot(ct_gts, 'b+')
plt.legend(['ct_preds', 'ct_gts'])
plt.title('Pred vs GT')
plt.show()
error = np.array(ct_preds) - np.array(ct_gts)
plt.plot(error)
plt.title('Pred - GT, mean = {}, MAE={}'.format(
    str(round(np.mean(error), 3)),
    str(round(np.mean(np.abs(error)), 3))
))
plt.show()
idx_max_error = np.argsort(np.abs(error))[::-1]

# Show the 5 worst samples
for worst_idx in idx_max_error[:4].tolist() + [dis_idx]:
    test_x_display = np.squeeze(test_x[worst_idx])
    test_y_display = np.squeeze(test_y[worst_idx])
    path_test_display = test_img_paths[worst_idx]
    pred = np.squeeze(model.predict(np.expand_dims(test_x_display, axis=0)))
    fg, (ax_x_ori, ax_y, ax_pred) = plt.subplots(1, 3, figsize=(20, 4))
    ax_x_ori.imshow(cv2.cvtColor(cv2.imread(path_test_display), cv2.COLOR_BGR2RGB))
    #ax_x_ori = cv2.resize(ax_x_ori, (int(ax_x_ori.shape[1] / 8), int(ax_x_ori.shape[0] / 8)), interpolation=cv2.INTER_CUBIC)
    ax_x_ori.set_title('Original Image')
    ax_y.imshow(test_y_display, cmap=plt.cm.jet)
    ax_y.set_title('Ground_truth: ' + str(np.sum(test_y_display)))
    ax_pred.imshow(pred, cmap=plt.cm.jet)
    ax_pred.set_title('Prediction: ' + str(np.sum(pred)))
    plt.show()