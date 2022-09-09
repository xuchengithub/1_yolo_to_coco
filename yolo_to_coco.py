# This file converts object detection annotations from YOLO format (.txt and .names) to COCO format (.json)
# The base code is originally from: https://www.programmersought.com/article/76707275021/
# It was modified extensively as the original code was not working for my needs.
# More information about the conversion process can be found here: https://prabhjotkaurgosal.com/weekly-learning-blogs/
import os
import json
import cv2
import random
import time
# (OPTIONAL) step
# Preparing the Dataset
# Convert all images to .jpg (This is not necessary but made my life easier down the line)

from PIL import Image
#--------------------------------------------------------------------------------------------------------------
# Category file, one category per line
yolo_format_classes_path = '/usr/src/app/csv_to_coco/labels.txt'
# Write the category according to your own data set. 

#Read the categories file and extract all categories
with open(yolo_format_classes_path,'r') as f1:
    lines1 = f1.readlines()
categories = []
for j,label in enumerate(lines1):
    label = label.strip()
    categories.append({'id':j+1,'name':label,'supercategory': label})
    
write_json_context = dict()
write_json_context['info'] = {'description': '', 'url': '', 'version': '', 'year': 2021, 'contributor': '', 'date_created': '2021-02-12 11:00:08.5'}
write_json_context['licenses'] = [{'id': 1, 'name': None, 'url': None}]
write_json_context['categories'] = categories
write_json_context['images'] = []
write_json_context['annotations'] = []
# Read the YOLO formatted label files (.txt) to extarct bounding box information and store in COCO format

#Read the label files (.txt) to extarct bounding box information and store in COCO format
directory_labels = os.fsencode("/usr/src/app/csv_to_coco/yolo")
#directory_images = os.fsencode("/home/jyoti/Desktop/csc8800/datasets/DoorDetectDataset/test")
directory_images = os.fsencode("/usr/src/app/csv_to_coco/yolo_images")

file_number = 1
num_bboxes = 1
for file in os.listdir(directory_images):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"):
        img_path = (os.path.join(directory_images.decode("utf-8"), filename))
        base=os.path.basename(img_path)
        file_name_without_ext = os.path.splitext(base)[0] # name of the file without the extension
        yolo_annotation_path  = os.path.join(directory_labels.decode("utf-8"), file_name_without_ext+ "." + 'txt')
        img_name = os.path.basename(img_path) # name of the file without the extension
        img_context = {}
        height,width = cv2.imread(img_path).shape[:2]
        img_context['file_name'] = img_name
        img_context['height'] = height
        img_context['width'] = width
        img_context['date_captured'] = '2021-02-12 11:00:08.5'
        img_context['id'] = file_number # image id
        img_context['license'] = 1
        img_context['coco_url'] =''
        img_context['flickr_url'] = ''
        write_json_context['images'].append(img_context)
        
        with open(yolo_annotation_path,'r') as f2:
            lines2 = f2.readlines() 

        for i,line in enumerate(lines2): # for loop runs for number of annotations labelled in an image
            line = line.split(' ')
            bbox_dict = {}
            class_id, x_yolo,y_yolo,width_yolo,height_yolo= line[0:]
            x_yolo,y_yolo,width_yolo,height_yolo,class_id= float(x_yolo),float(y_yolo),float(width_yolo),float(height_yolo),int(class_id)
            bbox_dict['id'] = num_bboxes
            bbox_dict['image_id'] = file_number
            bbox_dict['category_id'] = class_id+1
            bbox_dict['iscrowd'] = 0 # There is an explanation before
            h,w = abs(height_yolo*height),abs(width_yolo*width)
            bbox_dict['area']  = h * w
            x_coco = round(x_yolo*width -(w/2))
            y_coco = round(y_yolo*height -(h/2))
            if x_coco <0: #check if x_coco extends out of the image boundaries
                x_coco = 1
            if y_coco <0: #check if y_coco extends out of the image boundaries
                y_coco = 1
            bbox_dict['bbox'] = [x_coco,y_coco,w,h]
            bbox_dict['segmentation'] = [[x_coco,y_coco,x_coco+w,y_coco, x_coco+w, y_coco+h, x_coco, y_coco+h]]
            write_json_context['annotations'].append(bbox_dict)
            num_bboxes+=1
        
        file_number = file_number+1
        continue
    else:
        continue
        
 # Finally done, save!
#coco_format_save_path = '/home/jyoti/Desktop/csc8800/datasets/DoorDetectDataset/test.json'
coco_format_save_path = '/usr/src/app/csv_to_coco/train.json'
with open(coco_format_save_path,'w') as fw:
    json.dump(write_json_context,fw)
