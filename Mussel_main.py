import os
import cv2
import time
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from utils_gen import gen_paths_img_dm, gen_var_from_paths
from utils_imgproc import norm_by_imagenet
#%matplotlib inline
plt.ioff()

from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
#change_lr = LearningRateScheduler(step_decay)


# Settings
net = 'countingnet'
dataset = "A"
# Generate paths of (train, test) x (img, dm)
(test_img_paths, train_img_paths), (test_dm_paths, train_dm_paths) = gen_paths_img_dm(
    path_file_root='data/mussel300/',
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
#print(train_x)
print(type(train_x))
print("train_x.shape", train_x.shape)
print("train_y.shape", train_y.shape)
print('Train data size:', train_x.shape[0], train_y.shape[0], len(train_img_paths))
# Delete the directory for saving weights during last training.
weights_dir = 'data/mussel300/weights_' + dataset
if os.path.exists(weights_dir):
    shutil.rmtree(weights_dir)

from keras.optimizers import Adam, SGD, RMSprop
from keras.utils import plot_model
from countingnet import countingnet
from utils_loss import MSE_BCE


# Create empty directory for saving weights during training
if os.path.exists(weights_dir):
    shutil.rmtree(weights_dir)
os.makedirs(weights_dir)

# Settings of network
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
LOSS = 'MSE_BCE'
#optimizer = Adam(lr=1e-3)
optimizer = Adam(lr=1e-4, decay=5e-4)
#optimizer = RMSprop(lr=1e-4, decay=5e-3)
#optimizer = SGD(lr=1e-4, decay=5e-3, nesterov=False)

# Create my model
model = countingnet(input_shape=(None, None, 3))
model.compile(optimizer=optimizer, loss=MSE_BCE)
model.summary()
if not os.path.exists('models'):
    os.makedirs('models')
#plot_model(model, 'models/{}.png'.format(net))
with open('data/mussel300/models/{}.json'.format(net), 'w') as fout:
    fout.write(model.to_json())

import sys
import pandas as pd
from time import time, ctime
from utils_imgproc import image_preprocessing
from utils_callback import eval_loss, callbacks_during_train


# Settings of training
batch_size = 8
epoch = 600
val_rate = 0.5
val_rate_dec = {'A': [14, 5], 'B': [9, 8.5]}
len_train = train_x.shape[0]
num_iter = int((len_train-0.1) // batch_size + 1)
best_values = {'mae': 1e5, 'rmse': 1e5, 'sfn': 1e5, 'mape': 1e5}
losses = [[1e5, 1e5, 1e5, 1e5]]
# Settings of display
dis_idx = 16 if dataset == 'B' else 0
dis_path = test_img_paths[dis_idx]
dis_x = test_x[dis_idx]
dis_y = test_y[dis_idx]
dis_lim = (0, 35) if dataset == 'B' else (40, 150)
time_st = time()

# Training iterations
for ep in range(epoch):
    print(ep)

    state = np.random.get_state()
    np.random.shuffle(train_x)

    np.random.set_state(state)
    np.random.shuffle(train_y)

    print("shuffled!")
    #change_lr = LearningRateScheduler(step_decay(ep))
    for idx_train in range(0, len_train, batch_size):
        #print("idx_train:", idx_train)
        dis_epoch = str(ep+1)+'-'+str(idx_train+1)+'_'+str(len_train)
        x, y = train_x[idx_train:idx_train+batch_size], train_y[idx_train:idx_train+batch_size]
        print("x.shape:", x.shape)
        print("y.shape:", y.shape)
        # Preprocessings on raw images
        x, y = image_preprocessing(x, y, flip_hor=True, brightness_shift=False)
        model.fit(x, y, batch_size=8 ,verbose=0)
        idx_val = (idx_train / batch_size + 1)
        # Eval losses and save models
        if idx_val % (num_iter * val_rate) == 0:
            # To see predictions during training in directory 'tmp'
#             callbacks_during_train(
#                 model, dis_x=dis_x, dis_y=dis_y, dis_path=dis_path,
#                 net=net, epoch=dis_epoch
#             )
            #to_save = False
            loss = eval_loss(model, test_x, test_y, quality=False)
            print("loss:", loss)
            if loss[0] < val_rate_dec[dataset][0]:
                val_rate = 0.25
            if loss[0] < val_rate_dec[dataset][1]:
                val_rate = 0.1
            losses.append(loss)
            if (loss[0] < best_values['mae']) or (loss[0] == best_values['mae'] and loss[1] < best_values['rmse']):
                model.save_weights(os.path.join(weights_dir, '{}_best.hdf5'.format(net)))
            for idx_best in range(len(loss)):
                if loss[idx_best] < best_values[list(best_values.keys())[idx_best]]:
                    best_values[list(best_values.keys())[idx_best]] = loss[idx_best]
                    to_save = True
            if to_save:
                path_save = os.path.join(weights_dir, ''.join([
                    net,
                    '_MAE', str(round(loss[0], 3)), '_RMSE', str(round(loss[1], 3)),
                    '_SFN', str(round(loss[2], 3)), '_MAPE', str(round(loss[3], 3)),
                    '_epoch', str(ep+1), '-', str(idx_val), '.hdf5'
                ]))
                model.save_weights(path_save)
                to_save = False
            print('Change In epoch {}, with MAE-RMSE-SFN-MAPE={}, time consuming={}m-{}s\r'.format(
                dis_epoch, np.round(np.array(losses)[-1, :], 2),
                int(time_consuming / 60), int(time_consuming - int(time_consuming / 60) * 60)
            ))
        # Progress panel
        time_consuming = time() - time_st
        sys.stdout.write('In epoch {}, with MAE-RMSE-SFN-MAPE={}, time consuming={}m-{}s\r'.format(
            dis_epoch, np.round(np.array(losses)[-1, :], 2),
            int(time_consuming/60), int(time_consuming-int(time_consuming/60)*60)
        ))
        print('In epoch {}, with MAE-RMSE-SFN-MAPE={}, time consuming={}m-{}s\r'.format(
            dis_epoch, np.round(np.array(losses)[-1, :], 2),
            int(time_consuming/60), int(time_consuming-int(time_consuming/60)*60)
        ))
        sys.stdout.flush()

# Save records
losses = np.array(losses[1:])
pd.DataFrame(losses).to_csv('{}/loss.csv'.format(weights_dir), index=False, header=['MAE', 'RMSE', 'SFN', 'MAPE'])
losses_MAE, losses_RMSE, losses_SFN, losses_MAPE = losses[:, 0], losses[:, 1], losses[:, 2], losses[:, 3]
plt.plot(losses_MAE, 'r')
plt.plot(losses_RMSE, 'b')
multiplier = int(round(dis_lim[1] / np.max(losses_SFN)))
plt.plot(losses_SFN * multiplier, 'g')
plt.legend(['MAE', 'RMSE', 'SFN*{}'.format(multiplier)])
plt.ylim(dis_lim)
plt.title('Val_losses in {} epochs'.format(epoch))
plt.savefig('{}/{}_val_loss.png'.format(weights_dir, net))
plt.show()

# Rename weights_dir by the trainging end time, to prevent the careless deletion or overwriting
end_time_of_train = '-'.join(ctime().split()[:-2])
suffix_new_dir = '_{}_{}_bestMAE{}_{}'.format(dataset, LOSS, str(round(best_values['mae'], 3)), end_time_of_train)
weights_dir_neo = 'weights'+suffix_new_dir
shutil.move('data/mussel300/weights_{}'.format(dataset), weights_dir_neo)
print("weights_dir_neo:", weights_dir_neo)
print("DONE!!!!!!!!!!!!!!!!!!!")

# # Analysis on results
# # dis_idx = 16 if dataset == 'B' else 0
# # weights_dir_neo = 'weights_B_MSE_BCE_bestMAE7.846_Sat-May-18'
# model = model_from_json(open('data/Mussel/models/{}.json'.format(net), 'r').read())
# model.load_weights(os.path.join(weights_dir_neo, '{}_best.hdf5'.format(net)))
# ct_preds = []
# ct_gts = []
# for i in range(len(test_x[:])):
#     if i % 100 == 0:
#         print('{}/{}'.format(i, len(test_x)))
#     i += 0
#     test_x_display = np.squeeze(test_x[i])
#     test_y_display = np.squeeze(test_y[i])
#     path_test_display = test_img_paths[i]
#     pred = np.squeeze(model.predict(np.expand_dims(test_x_display, axis=0)))
#     ct_pred = np.sum(pred)
#     ct_gt = np.sum(test_y_display)
#     ct_preds.append(ct_pred)
#     ct_gts.append(ct_gt)
# plt.plot(ct_preds, 'r>')
# plt.plot(ct_gts, 'b+')
# plt.legend(['ct_preds', 'ct_gts'])
# plt.title('Pred vs GT')
# plt.show()
# error = np.array(ct_preds) - np.array(ct_gts)
# plt.plot(error)
# plt.title('Pred - GT, mean = {}, MAE={}'.format(
#     str(round(np.mean(error), 3)),
#     str(round(np.mean(np.abs(error)), 3))
# ))
# plt.show()
# idx_max_error = np.argsort(np.abs(error))[::-1]
#
# # Show the 5 worst samples
# for worst_idx in idx_max_error[:14].tolist() + [dis_idx]:
#     test_x_display = np.squeeze(test_x[worst_idx])
#     test_y_display = np.squeeze(test_y[worst_idx])
#     path_test_display = test_img_paths[worst_idx]
#     pred = np.squeeze(model.predict(np.expand_dims(test_x_display, axis=0)))
#     fg, (ax_x_ori, ax_y, ax_pred) = plt.subplots(1, 3, figsize=(20, 4))
#     ax_x_ori.imshow(cv2.cvtColor(cv2.imread(path_test_display), cv2.COLOR_BGR2RGB))
#     ax_x_ori.set_title('Original Image')
#     ax_y.imshow(test_y_display, cmap=plt.cm.jet)
#     ax_y.set_title('Ground_truth: ' + str(np.sum(test_y_display)))
#     ax_pred.imshow(pred, cmap=plt.cm.jet)
#     ax_pred.set_title('Prediction: ' + str(np.sum(pred)))
#     plt.show()
