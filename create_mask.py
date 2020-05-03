import os
from PIL import Image
import cv2
import tensorflow as tf 
import numpy as np
import copy
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

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

#Dataset
TEST_IMAGE_PATHS = "datasets/tommaso/all"
nums = os.listdir(TEST_IMAGE_PATHS + "/depth")
nums = [x[:-4 ] for x in nums]
done_so_far = os.listdir(TEST_IMAGE_PATHS + "/mask_real")
done_so_far = [x[:-4 ] for x in done_so_far]
counter = 0
while True:
    idx = nums[counter]
    if idx in done_so_far:
        counter = counter + 1
        continue
    image_np = Image.open(TEST_IMAGE_PATHS+"/JPEGImages/"+ str(idx) + ".jpg")
    image_np = np.array(image_np)
    counter = counter + 1
    image_np_2 =copy.deepcopy(image_np)
    image_np = np.array(image_np)

    if True:
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
    else: 
        mask = Image.open(TEST_IMAGE_PATHS+"/mask/"+ str(idx) + ".png")

    cv2.imwrite(TEST_IMAGE_PATHS+"/mask_real/"+ str(idx) + ".png", mask)
    continue