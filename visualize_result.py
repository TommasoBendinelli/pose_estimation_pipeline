import os 
import sys 
from os.path import dirname 
sys.path.append("tools/")
import random
import numpy as np
import cv2
import yaml
import pickle
import copy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from lib.knn.__init__ import KNearestNeighbor
from tools.visualization import Visualizer
from datasets.tommaso.dataset import PoseDataset as Tommaso_poseDataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("mine", help="Which dataset to play? ", type=int)
parser.add_argument("also testing", help="Plot the results of the testing or just the training", type=int)
args = vars(parser.parse_args())
mine = args['mine'] 
also_testing = args['also testing']
 
#Select which dataset to visualize
if not mine:
    opt_dataset_root = "./datasets/linemod/Linemod_preprocessed"
    opt_model = "trained_models/linemod/pose_model_9_0.012956139583687484.pth"
    opt_refine_model = "trained_models/linemod/pose_refine_model_95_0.007274364822843561.pth"

    num_objects = 13
    objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
    num_points = 500
    iteration = 4
    bs = 1
    dataset_config_dir = 'datasets/linemod/dataset_config'
    output_result_dir = 'experiments/eval_result/linemod'
    knn = KNearestNeighbor(1)

if mine:
    opt_dataset_root = "./datasets/tommaso/tommaso_preprocessed"
    opt_model = "trained_models/tommaso/pose_model_1_0.021815784575678215.pth"
    opt_refine_model = "trained_models/tommaso/pose_refine_model_10_0.013444575836113132.pth"
    #opt_model = "trained_models/tommaso/pose_model_125_0.04890690553695597.pth"
    #opt_refine_model = "trained_models/tommaso/pose_refine_model_8_0.048650759212831234.pth"

    num_objects = 1
    objlist = [3]
    num_points = 500
    iteration = 4
    bs = 1
    dataset_config_dir = 'datasets/linemod/dataset_config'
    output_result_dir = 'experiments/eval_result/linemod'
    knn = KNearestNeighbor(1)

#Defining neural Network
# if  also_testing:
estimator = PoseNet(num_points = num_points, num_obj = num_objects)
estimator.cuda()
refiner = PoseRefineNet(num_points = num_points, num_obj = num_objects)
refiner.cuda()
estimator.load_state_dict(torch.load(opt_model))
refiner.load_state_dict(torch.load(opt_refine_model))
estimator.eval()
refiner.eval()



if not mine:
    #Defining camera
    cam_cx = 325.26110
    cam_cy = 242.04899
    cam_fx = 572.41140
    cam_fy = 573.57043
    K = np.array([[cam_fx,0,cam_cx],[0,cam_fy,cam_cy],[0,0,1]])

    with open('testdataset_eval.pkl', 'rb') as input:
        testdataset = pickle.load(input)

if mine: 
    cam_fx = 614.28125
    cam_fy = 614.4807739257812
    cam_cx = 323.3623962402344
    cam_cy = 247.32833862304688
    K = np.array([[cam_fx,0,cam_cx],[0,cam_fy,cam_cy],[0,0,1]])
    testdataset = Tommaso_poseDataset('eval', num_points, False, opt_dataset_root, 0.0, True, is_visualized = True)


testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=0) #In visualization is fine if it is 0, in training no

sym_list = testdataset.get_sym_list()
num_points_mesh = testdataset.get_num_points_mesh()
criterion = Loss(num_points_mesh, sym_list)
criterion_refine = Loss_refine(num_points_mesh, sym_list)

diameter = []
meta_file = open('{0}/models_info.yml'.format(dataset_config_dir), 'r')
meta = yaml.load(meta_file)
for obj in objlist:
    diameter.append(meta[obj]['diameter'] / 1000.0 * 0.1)
print(num_objects)

success_count = [0 for i in range(num_objects)]
num_count = [0 for i in range(num_objects)]
#fw = open('{0}/eval_result_logs.txt'.format(output_result_dir), 'w')

# test1 = Visualizer(testdataset)
# cam_cx = 325.26110
# cam_cy = 242.04899
# cam_fx = 572.41140
# cam_fy = 573.57043
# K = np.array([[cam_fx,0,cam_cx],[0,cam_fy,cam_cy],[0,0,1]])


for i, data in enumerate(testdataloader, 0):
    # FIX ME for making it work
    try: 
        points, choose, img, target, model_points, idx, ori_img, img_masked, index, Candidate_mask, box, T_truth  = data
    except Exception as e:
        print(e)
        continue

    if len(points.size()) == 2:
        print('No.{0} NOT Pass! Lost detection!'.format(i))
        fw.write('No.{0} NOT Pass! Lost detection!\n'.format(i))
        continue
    points, choose, img, model_points, idx = Variable(points).cuda(), \
                                                     Variable(choose).cuda(), \
                                                     Variable(img).cuda(), \
                                                     Variable(model_points).cuda(), \
                                                     Variable(idx).cuda()

    
    pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
    pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
    pred_c = pred_c.view(bs, num_points)
    how_max, which_max = torch.max(pred_c, 1)
    pred_t = pred_t.view(bs * num_points, 1, 3)

    my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
    my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
    my_pred = np.append(my_r, my_t)

    for ite in range(0, iteration):
        T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
        my_mat = quaternion_matrix(my_r)
        R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
        my_mat[0:3, 3] = my_t
        
        new_points = torch.bmm((points - T), R).contiguous()
        pred_r, pred_t = refiner(new_points, emb, idx)
        pred_r = pred_r.view(1, 1, -1)
        pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
        my_r_2 = pred_r.view(-1).cpu().data.numpy()
        my_t_2 = pred_t.view(-1).cpu().data.numpy()
        my_mat_2 = quaternion_matrix(my_r_2)
        my_mat_2[0:3, 3] = my_t_2

        my_mat_final = np.dot(my_mat, my_mat_2)
        my_r_final = copy.deepcopy(my_mat_final)
        my_r_final[0:3, 3] = 0
        my_r_final = quaternion_from_matrix(my_r_final, True)
        my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

        my_pred = np.append(my_r_final, my_t_final)
        my_r = my_r_final
        my_t = my_t_final

    # Here 'my_pred' is the final pose estimation result after refinement ('my_r': quaternion, 'my_t': translation)

    model_points = model_points[0].cpu().detach().numpy()
    my_r = quaternion_matrix(my_r)[:3, :3]
    if mine: 
        pred = np.dot(model_points, my_r.T) + my_t
    if not mine:
        pred = np.dot(model_points, my_r.T) + my_t
    point_proj = np.dot(K,pred.T)
    points_2d = point_proj.T[:,[0,1]]/point_proj.T[:,[2]]
    points_2d = np.floor(points_2d).astype(int)

    target = np.squeeze(target.numpy())
    target_proj = np.dot(K,target.T)
    targets_2d = target_proj.T[:,[0,1]]/target_proj.T[:,[2]]
    targets_2d = np.floor(targets_2d).astype(int)
    img = np.squeeze(ori_img.numpy())
    np.testing.assert_equal(img,testdataset[index][6])
    print(index)
    
    y_min, y_max, x_min, x_max = list(map(lambda x: x.numpy().item(),box))
    cv2.line(img,(x_min,y_max),(x_max,y_max),(0,0,255),3)
    cv2.line(img,(x_max,y_max),(x_max,y_min),(0,0,255),3)
    cv2.line(img,(x_min,y_min),(x_min,y_max),(0,0,255),3)
    cv2.line(img,(x_max,y_min),(x_min,y_min),(0,0,255),3)


    #This represents what the network predicts
    if also_testing:
        for pt in points_2d:
            pt = tuple(pt)
            cv2.circle(img,pt,1,[0,0,255],1)



    Candidate_mask = np.squeeze(Candidate_mask.numpy()).astype(int)

    # #This represents the points sampled from the mask (a grid has been created from where points are sampled)
    # for mask in Candidate_mask:
    #     mask = tuple(mask)
    #     cv2.circle(img,mask,1,[0,255,0],1)

    # #This represents the points computed through the ground truth transformation
    # for tg in targets_2d:
    #     tg = tuple(tg)
    #     cv2.circle(img,tg,1,[255,0,0],1)

    #test1 = np.squeeze(img_masked.numpy())
    cv2.imshow("idx",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        


