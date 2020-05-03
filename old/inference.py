
# def load_model(model_name):
#   base_url = 'http://download.tensorflow.org/models/object_detection/'
#   model_file = model_name + '.tar.gz'
#   model_dir = tf.keras.utils.get_file(
#     fname=model_name, 
#     origin=base_url + model_file,
#     untar=True)

#   model_dir = pathlib.Path(model_dir)/"saved_model"

#   model = tf.saved_model.load(str(model_dir))
#   model = model.signatures['serving_default']

#   return model
from PIL import Image
import cv2
import os 
import numpy as np
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import tensorflow as tf 
import os
import numpy.ma as ma
import os 
import sys 
from os.path import dirname 
sys.path.append("tools/")
import random
import numpy as np
import cv2
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
# from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from lib.knn.__init__ import KNearestNeighbor
import torchvision.transforms as transforms



# from tools.visualization import Visualizer
#from models.dataset import PoseDataset as Tommaso_poseDataset
import argparse
import open3d as o3d




def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  output_dict = model(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict

def show_inference(model, image_path):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = np.array(Image.open(image_path))
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)

    display(Image.fromarray(image_np))


    return model

def load_image_into_numpy(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict1 = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict = {}
            output_dict['num_detections'] = int(output_dict1['num_detections'][0])
            output_dict['detection_classes'] = output_dict1[
                    'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict1['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict1['detection_scores'][0]
            if 'detection_masks' in output_dict1:
                output_dict['detection_masks'] = output_dict1['detection_masks'][0]
    return output_dict


xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
def preprocessing_for_pose_estimation(rgb_image=None, depth_img=None, label=None, mesh=None, num_points=None):
    #img = Image.open(self.list_rgb[index])
    #depth = np.array(Image.open(self.list_depth[index]))
    # obj = self.list_obj[index]
    # rank = self.list_rank[index]        

    mask_depth = ma.getmaskarray(ma.masked_not_equal(depth_img, 0))
    mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))
    # else:  #FIX ME PLEASE!!!
    #     mask_label = ma.getmaskarray(ma.masked_equal(label, np.array([255, 255, 255])))[:, :, 0]
    
    mask = mask_label * mask_depth

    # if self.add_noise:
    #     img = self.trancolor(img)

    img = np.array(rgb_image)[:, :, :3]
    img = np.transpose(img, (2, 0, 1))
    img_masked = img

    rmin, rmax, cmin, cmax = get_bbox(mask_to_bbox(mask_label))

    img_masked = img_masked[:, rmin:rmax, cmin:cmax]
    #p_img = np.transpose(img_masked, (1, 2, 0))
    #scipy.misc.imsave('evaluation_result/{0}_input.png'.format(index), p_img)

    # target_r = np.resize(np.array(meta['cam_R_m2c']), (3, 3))
    # target_t = np.array(meta['cam_t_m2c'])
    # add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])

    choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
    if len(choose) == 0:
        cc = torch.LongTensor([0])
        return(cc, cc, cc, cc, cc, cc)

    if len(choose) > num_points:
        c_mask = np.zeros(len(choose), dtype=int)
        c_mask[:num_points] = 1
        np.random.shuffle(c_mask)
        choose = choose[c_mask.nonzero()]
    else:
        choose = np.pad(choose, (0,num_points - len(choose)), 'wrap')
    
    depth_masked = depth_img[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)

    Candidate_mask = np.concatenate((ymap_masked,xmap_masked),axis=1)
    choose = np.array([choose])

    cam_scale = 1.0
    pt2 = depth_masked / cam_scale
    pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
    pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
    cloud = np.concatenate((pt0, pt1, pt2), axis=1)
    cloud = cloud / 1000.0

    #fw = open('evaluation_result/{0}_cld.xyz'.format(index), 'w')
    #for it in cloud:
    #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
    #fw.close()

    
    # dellist = [j for j in range(0, len(model_points))]
    # dellist = random.sample(dellist, len(model_points) - self.num_pt_mesh_small)
    # model_points = np.delete(model_points, dellist, axis=0)

    #fw = open('evaluation_result/{0}_model_points.xyz'.format(index), 'w')
    #for it in model_points:
    #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
    #fw.close()

    # target_t_re = target_t.reshape(3,1)
    # tmp = np.concatenate((target_r,target_t_re),axis=1)
    # T = np.concatenate((tmp,[[0,0,0,1]]),axis=0)
    model_points = mesh.sample_points_poisson_disk(num_points)
    # target = copy.deepcopy(model_points)
    # target.transform(T)
    # target = np.asarray(target.points)
    model_points = np.asarray(model_points.points)

    # samples =self.mesh[obj].sample(self.num)
    # mesh = self.mesh[obj].copy()
    # mesh.apply_transform(T)
    # samples_corr = mesh.sample(self.num)
    

    #target = np.dot(model_points, target_r.T)
    # if self.add_noise:
    #     target = np.add(target, target_t+ add_t)
    #     out_t = target_t + add_t
    # else:
    #     target = np.add(target, target_t)
    #     out_t = target_t 

    #fw = open('evaluation_result/{0}_tar.xyz'.format(index), 'w')
    #for it in target:
    #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
    #fw.close()
    return {"depth_cloud": torch.from_numpy(cloud.astype(np.float32)).unsqueeze_(0), \
        "candidate_area": torch.LongTensor(choose.astype(np.int32)).unsqueeze_(0), \
        "img_masked": norm(torch.from_numpy(img_masked.astype(np.float32))).unsqueeze_(0), \
        "model_points": torch.from_numpy(model_points.astype(np.float32)).unsqueeze_(0), \
        "obj": torch.LongTensor([0]).unsqueeze_(0)}

def get_bbox(bbox):
    border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 480:
        bbx[1] = 479
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 640:
        bbx[3] = 639                
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > 480:
        delt = rmax - 480
        rmax = 480
        rmin -= delt
    if cmax > 640:
        delt = cmax - 640
        cmax = 640
        cmin -= delt
    return rmin, rmax, cmin, cmax

def mask_to_bbox(mask):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x = 0
    y = 0
    w = 0
    h = 0
    for contour in contours:
        tmp_x, tmp_y, tmp_w, tmp_h = cv2.boundingRect(contour)
        if tmp_w * tmp_h > w * h:
            x = tmp_x
            y = tmp_y
            w = tmp_w
            h = tmp_h
    return [x, y, w, h]


PATH_TO_FROZEN_GRAPH = "models/inference_graph_mask/frozen_inference_graph.pb"
PATH_TO_LABELS = "datasets/tommaso/tommaso_label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
 
opt_dataset_root = "datasets/tommaso/"
opt_model = "models/pose/pose_model_1_0.005381468217819929"
opt_refine_model = "models/pose/pose_refine_model_2_0.004954009364381983.pth"
#opt_model = "trained_models/tommaso/pose_model_125_0.04890690553695597.pth"
#opt_refine_model = "trained_models/tommaso/pose_refine_model_8_0.048650759212831234.pth"

cam_fx = 614.28125
cam_fy = 614.4807739257812
cam_cx = 323.3623962402344
cam_cy = 247.32833862304688
K = np.array([[cam_fx,0,cam_cx],[0,cam_fy,cam_cy],[0,0,1]])
#testdataset = Tommaso_poseDataset('eval', num_points, False, opt_dataset_root, 0.0, True, is_visualized = True)

num_objects = 1
objlist = [3]
num_points = 500
iteration = 0
bs = 1
#dataset_config_dir = 'datasets/linemod/dataset_config'
#output_result_dir = 'experiments/eval_result/linemod'
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

#Dataset
TEST_IMAGE_PATHS = "datasets/tommaso" 

#Loading model object
mesh = o3d.io.read_triangle_mesh(TEST_IMAGE_PATHS + "/" + "models/obj_03.ply")


for image_path in os.listdir(TEST_IMAGE_PATHS + "/rgb/" ):
    if image_path[-3:] == "jpg":
        image_name = image_path[:-4]
        abs_image_path = Image.open(TEST_IMAGE_PATHS+ "/rgb/" + image_name + ".jpg")
        depth_image_path = Image.open(TEST_IMAGE_PATHS+"/depth/"+image_name + ".png")
        # depth_image_path = Image.open(TEST_IMAGE_PATHS+"/depth_test/"+ "20" + ".png")
        # rgb_image_path = Image.open(TEST_IMAGE_PATHS+"/JPEGImages_test/"+ "20" + ".jpg")
        depth = np.array(depth_image_path)
        #image_path.show()
        image_np = load_image_into_numpy(rgb_image_path)
        #image_np_expanded = np.expand_dims(image_np, axis=0)
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=2)
        # cv2.imshow("Masked", image_np)
        # k = 0xFF & cv2.waitKey(1)
        mask = output_dict.get('detection_masks')[0]*255
        #cv2.imwrite("datasets/tommaso/test_mask/" + image_name + ".png", mask) 
        preprocessed = preprocessing_for_pose_estimation(image_np, depth, mask, mesh,500)
        points, choose, img, model_points, idx = Variable( preprocessed["depth_cloud"]).cuda(), \
                                                     Variable(preprocessed['candidate_area']).cuda(), \
                                                     Variable(preprocessed['img_masked']).cuda(), \
                                                     Variable(preprocessed["model_points"]).cuda(), \
                                                     Variable( preprocessed["obj"]).cuda()
        pred_r, pred_t, pred_c, emb = estimator(img, points, choose,idx)
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
    pred = np.dot(model_points, my_r.T) + my_t
    point_proj = np.dot(K,pred.T)
    points_2d = point_proj.T[:,[0,1]]/point_proj.T[:,[2]]
    points_2d = np.floor(points_2d).astype(int)

    # target = np.squeeze(target.numpy())
    # target_proj = np.dot(K,target.T)
    # targets_2d = target_proj.T[:,[0,1]]/target_proj.T[:,[2]]
    # targets_2d = np.floor(targets_2d).astype(int)
    # img = np.squeeze(ori_img.numpy())
    # np.testing.assert_equal(img,testdataset[index][6])
    # print(index)
    
    # y_min, y_max, x_min, x_max = list(map(lambda x: x.numpy().item(),box))
    # cv2.line(img,(x_min,y_max),(x_max,y_max),(0,0,255),3)
    # cv2.line(img,(x_max,y_max),(x_max,y_min),(0,0,255),3)
    # cv2.line(img,(x_min,y_min),(x_min,y_max),(0,0,255),3)
    # cv2.line(img,(x_max,y_min),(x_min,y_min),(0,0,255),3)


    #This represents what the network predicts
    for pt in points_2d:
        pt = tuple(pt)
        cv2.circle(image_np,pt,1,[0,0,255],1)



    # Candidate_mask = np.squeeze(Candidate_mask.numpy()).astype(int)

    # #This represents the points sampled from the mask (a grid has been created from where points are sampled)
    # for mask in Candidate_mask:
    #     mask = tuple(mask)
    #     cv2.circle(img,mask,1,[0,255,0],1)

    # #This represents the points computed through the ground truth transformation
    # for tg in targets_2d:
    #     tg = tuple(tg)
    #     cv2.circle(img,tg,1,[255,0,0],1)

    #test1 = np.squeeze(img_masked.numpy())
    cv2.imshow("idx",image_np)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        

            

        
        # img.show()





#latest = tf.train.latest_checkpoint("training/")

