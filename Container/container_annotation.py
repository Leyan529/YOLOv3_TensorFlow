import xml.etree.ElementTree as ET
import os
from os import listdir, getcwd
from os.path import join
import shutil
import numpy as np


classes = ["aquarium","bottle","bowl","box","bucket","plastic_bag","plate",
"styrofoam","tire","toilet","tub","washing_machine","water_tower"]


def convert(n,size, box):
    image_w = size[0]
    image_h = size[1]

    dw = 1./(image_w)
    dh = 1./(image_h)
    x = (box[0] + (box[1] - box[0])/2.0) * 1.0 * dw
    y = (box[2] + (box[3] - box[2])/2.0) * 1.0 * dh
    w = (box[1] - box[0])*1.0 * dw
    h = (box[3] - box[2])*1.0 * dh    
    return (n,x, y, w, h)

def convert_annotation(path,image_id, train_file):
    in_file = open('./train_cdc/train_annotations/%s.xml'%(image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()
    width = int(root.find('size').findtext('width'))
    height = int(root.find('size').findtext('height'))

    ehance_fn_list=["","_bri1","_bri2","_col1","_col2","_con1","_con2","_sha1","_sha2"]
    for idx,d in enumerate(ehance_fn_list):        
        if root.find('object'):
            new_path = path.replace(".jpg",d+".jpg")
            train_file.write(new_path)               
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                continue
            cls_id = int(classes.index(cls))
            #xmlbox = obj.find('bndbox')

            xmin = obj.find('bndbox').findtext('xmin')
            ymin = obj.find('bndbox').findtext('ymin')
            xmax = obj.find('bndbox').findtext('xmax')
            ymax = obj.find('bndbox').findtext('ymax')
            b = [cls_id,xmin,ymin,xmax,ymax]       
            
            train_file.write(" " + " ".join([str(a) for a in b]))  
        if idx != len(ehance_fn_list)-1 :
            train_file.write("\n")

        # b = (float(xmin), float(xmax), float(ymin), float(ymax))
        # bb = convert(cls_id,(width, height), b)
        # label_file.write(" ".join([str(a) for a in bb]) + '\n')
        # print("name:{} xmin:{} ymin:{} xmax:{} ymax:{}".format(cls,bb[0],bb[1],bb[2],bb[3]))

        
def create_keras_data():
    train_XmlDir = "./train_cdc/train_annotations/"
    img_dir = "./train_cdc/train_images/"
    train_file = open('container.txt','w') 
    dir_data = listdir(train_XmlDir)
    for idx,f in enumerate(dir_data):    
        image_id = f.replace(".xml","")  
        img =  os.path.abspath(img_dir +'/%s.jpg'%(image_id))
        convert_annotation(img,image_id,train_file)
        if idx != (len(dir_data)-1):
            train_file.write("\n")
        
    train_file.close()
    print("Finish...")

def data_split(split,file,fn1,fn2):
    with open(file,"r") as f:
        lines = f.readlines()        
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*split)
    num_train = len(lines) - num_val
    train_lines = lines[:num_train]
    val_lines = lines[num_train:]

    with open(fn1,"w") as f:
        for idx,line in enumerate(train_lines):   
            f.write(str(idx) + " ")
            line = line.replace("\n","")     
            f.write(line)
            if idx != len(train_lines)-1:
                f.write("\n")
        f.flush()

    with open(fn2,"w") as f:
        for idx,line in enumerate(val_lines):
            f.write(str(idx) + " ")
            line = line.replace("\n","")   
            f.write(line)
            if idx != len(val_lines)-1:
                f.write("\n")
        f.flush()


    return train_lines , val_lines

if __name__ == '__main__':
    ### Some paths
    # train_file = './data/my_data/train.txt'  # The path of the training txt file.
    # val_file = './data/my_data/val.txt'  # The path of the validation txt file.
    train_file = "train.txt"  # The path of the training txt file.
    val_file = "val.txt"  # The path of the validation txt file.
    data_file = 'container.txt'  # The path of the validation txt file.
    create_keras_data()

    val_split = 0.1
    # train_img_cnt = len(open(train_file, 'r').readlines())
    # val_img_cnt = len(open(val_file, 'r').readlines())
    train_img , val_img = data_split(val_split,data_file,train_file,val_file)