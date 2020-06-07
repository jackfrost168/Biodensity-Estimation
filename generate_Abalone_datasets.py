import os
import cv2
import glob
import h5py
from scipy.io import loadmat
import numpy as np
from tqdm import tqdm
from utils_gen import gen_density_map_gaussian
import matplotlib.pyplot as plt

#%matplotlib inline


root = 'data/Abalone/'
part_A_train = os.path.join(root, 'train_data','images')
print(part_A_train)
part_A_test = os.path.join(root, 'test_data','images')
#part_B_train = os.path.join(root, 'part_B/train_data', 'images')
#part_B_test = os.path.join(root, 'part_B/test_data', 'images')
path_sets_A = [part_A_train, part_A_test]
#path_sets_B = [part_B_train, part_B_test]
img_paths_A = []
for path in path_sets_A:
    for img_path in glob.glob(os.path.join(path, '*.JPG')):
        img_paths_A.append(img_path)
print("img_paths_A len:",len(img_paths_A))
#img_paths_B = []
# for path in path_sets_B:
#     for img_path in glob.glob(os.path.join(path, '*.jpg')):
#         img_paths_B.append(img_path)
# print(len(img_paths_B))

# ####### Generate h5 to ground file  ##################
# for dataset in ['A']:
#     img_paths = eval('img_paths_'+dataset) #img_paths = img_paths_A
#     #print(img_paths)
#     for img_path in tqdm(img_paths):
#         print(img_path)
#         img_ori = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
#         #pts = loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
#         img = cv2.imread(img_path)
#         img = cv2.resize(img, (int(img.shape[1]/8), int(img.shape[0]/8)), interpolation=cv2.INTER_CUBIC)
#         sigma = 4  #if 'part_A' in img_path else 15
#         k = np.zeros((img.shape[0], img.shape[1]))
#         #gt = pts["image_info"][0, 0][0, 0][0]
#
#         import xml.dom.minidom as xmldom
#
#         # ratey = 4000 / 1152
#         # ratex = 6000 / 2048
#
#         xmlfilepath = img_path.replace('.JPG', '.xml').replace('images','annotation')
#         print(xmlfilepath)
#         domobj = xmldom.parse(xmlfilepath)
#         # print("xmldom.parse:", type(domobj))
#
#         elementobj = domobj.documentElement
#         # print("domobj.documentElement:", type(elementobj))
#         name = elementobj.getElementsByTagName("name")
#
#         subElementObj1 = elementobj.getElementsByTagName("xmin")
#         subElementObj2 = elementobj.getElementsByTagName("ymin")
#         subElementObj3 = elementobj.getElementsByTagName("xmax")
#         subElementObj4 = elementobj.getElementsByTagName("ymax")
#         list1 = []
#         list2 = []
#         center = []
#         for i in range(len(subElementObj1)):
#             if name[i].firstChild.data == "small":
#                 list1.append([subElementObj1[i].firstChild.data, subElementObj2[i].firstChild.data])
#                 list2.append([subElementObj3[i].firstChild.data, subElementObj4[i].firstChild.data])
#                 x1 = int(subElementObj1[i].firstChild.data)
#                 y1 = int(subElementObj2[i].firstChild.data)
#                 x2 = int(subElementObj3[i].firstChild.data)
#                 y2 = int(subElementObj4[i].firstChild.data)
#
#                 center.append([int((x1 + x2) / 2/8), int((y1 + y2) / 2/8)])
#
#         gt = center
#         print(len(gt))
#
#         for i in range(len(gt)):
#             if int(gt[i][1]/8) < img.shape[0] and int(gt[i][0]/8) < img.shape[1]:
#                 k[int(gt[i][1]/8), int(gt[i][0]/8)] = 1
#
#         DM = gen_density_map_gaussian(k, gt, sigma=sigma)
#         print('DM.shape:', DM.shape)
#         file_path = img_path.replace('.JPG', '.h5').replace('images', 'ground')
#         with h5py.File(file_path, 'w') as hf:
#             hf['density'] = DM

# Show a sample
img_paths = ['data/Abalone/train_data/images/DSC09648_3.JPG',
             'data/Abalone/train_data/images/DSC09764_5.JPG']
for img_path in img_paths:
    img_ori = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img_ori = cv2.resize(img_ori, (int(img_ori.shape[1] / 8), int(img_ori.shape[0] / 8)), interpolation=cv2.INTER_CUBIC)
    #pts = loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))

    img = cv2.imread(img_path)
    img = cv2.resize(img, (int(img.shape[1] / 8), int(img.shape[0] / 8)), interpolation=cv2.INTER_CUBIC)

    sigma = 4 #if 'part_A' in img_path else 15
    k = np.zeros((img.shape[0], img.shape[1]))

    import xml.dom.minidom as xmldom

    xmlfilepath = img_path.replace('.JPG', '.xml').replace('images','annotation')

    domobj = xmldom.parse(xmlfilepath)
    elementobj = domobj.documentElement
    name = elementobj.getElementsByTagName("name")

    subElementObj1 = elementobj.getElementsByTagName("xmin")
    subElementObj2 = elementobj.getElementsByTagName("ymin")
    subElementObj3 = elementobj.getElementsByTagName("xmax")
    subElementObj4 = elementobj.getElementsByTagName("ymax")
    list1 = []
    list2 = []
    center = []
    for i in range(len(subElementObj1)):
        if name[i].firstChild.data == "small":
            list1.append([subElementObj1[i].firstChild.data, subElementObj2[i].firstChild.data])
            list2.append([subElementObj3[i].firstChild.data, subElementObj4[i].firstChild.data])
            x1 = int(subElementObj1[i].firstChild.data)
            y1 = int(subElementObj2[i].firstChild.data)
            x2 = int(subElementObj3[i].firstChild.data)
            y2 = int(subElementObj4[i].firstChild.data)

            center.append([int((x1 + x2) / 2/8), int((y1 + y2) / 2/8)])

    gt = center
    print("gt len:", len(gt))

    for i in range(len(gt)):
        if int(gt[i][1]/8) < img.shape[0] and int(gt[i][0]/8) < img.shape[1]:
            k[int(gt[i][1]/8), int(gt[i][0]/8)] = 1

    DM = gen_density_map_gaussian(k, gt, sigma=sigma)
    print(DM.shape)

    fg, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 4))
    ax0.imshow(img_ori)
    ax0.set_title(str(np.squeeze(gt).shape[0]))
    ax1.imshow(np.squeeze(DM), cmap=plt.cm.jet)
    ax1.set_title('DM -- ' + str(np.sum(DM)))
    plt.show()

