import cv2
import subprocess
import shlex
import class_tracker
from sys import platform

# Pretrained classes in the model
classNames = {0: 'background',
              1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
              7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
              13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
              18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
              24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
              32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
              37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
              41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
              46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
              51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
              56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
              61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
              67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
              75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
              80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
              86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

# Dicionário contador
count_dict = {'cell phone' : 0, 'cup' : 0, 'bottle' : 0}

def id_class_name(class_id, classes):
    for key, value in classes.items():
        if class_id == key:
            return value

out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,480))

# Execute the argument passed in command line 
def python_to_bash(cli_args):
    args = shlex.split(cli_args)
    cli_input = subprocess.Popen(args, stdout = subprocess.PIPE)
    output, error = cli_input.communicate()
    return output.decode('utf-8'), error

# Instanciate the video object
if platform == 'linux':
    devices = 'v4l2-ctl --list-devices'
    terminal_output, erro = python_to_bash(devices)
    devices_list = terminal_output.split('\n')
    for n_line, text in enumerate(devices_list):
        if 'usb' in text:
            cam_dir = devices_list[n_line + 1].strip()
            break
    vobj = cv2.VideoCapture(int(cam_dir[-1]))
else:
    vobj = cv2.VideoCapture(0)

# Loading model
model = cv2.dnn.readNetFromTensorflow('models/frozen_inference_graph.pb',
                                      'models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

trk_gen = class_tracker.objects_updator(2)

def detection(image):
    dets = []
    class_list = []
    model.setInput(cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True))
    output = model.forward()

    for detection in output[0, 0, :, :]:
        confidence = detection[2]
        if confidence > .6:
            class_id = detection[1]
            class_name=id_class_name(class_id,classNames)
            if class_name in list(count_dict.keys()):
                box_x = detection[3] * image_width
                box_y = detection[4] * image_height
                box_width = detection[5] * image_width
                box_height = detection[6] * image_height
                dets.append((box_x, box_y, box_width - box_x, box_height - box_y))
                class_list.append(class_name)
                cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (0, 0, 255), thickness=3)
                cv2.putText(image,class_name ,(int(box_x + 5), int(box_y+.05*image_height)),cv2.FONT_HERSHEY_SIMPLEX,(.0015*image_width),(255, 0, 0), 2)

    new_idx = trk_gen.update_dets(image, dets)

    if len(new_idx) > 0:
        for idx in new_idx:
            count_dict[class_list[idx]] += 1

frames_count = 0

while True:
    status, image = vobj.read()
    image_height, image_width, _ = image.shape

    if frames_count == 0 or frames_count == 30:
        detection(image)
        frames_count = 0
    
    elif len(trk_gen.trackers_list) > 0:
        trk_gen.update_trks(image)
        

    # Writes the objects counter
    pos_y = 0.1
    for key in count_dict.keys():
        cv2.putText(image, key + ' :', (5, int(image_height * pos_y)), cv2.FONT_HERSHEY_SIMPLEX, (0.0015 * image_width), (0, 255, 0), 2)
        cv2.putText(image, str(count_dict[key]), (250, int(image_height * pos_y)), cv2.FONT_HERSHEY_SIMPLEX, (0.0015 * image_width), (0, 200, 128), 2)
        pos_y += 0.1

    cv2.imshow('image', image)
    out.write(image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frames_count += 1

cv2.destroyAllWindows()
out.release()
vobj.release()