import xml.etree.ElementTree as ET
import os
from os import listdir, getcwd
from os.path import join
import shutil



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

def convert_annotation(image_id, train_file):
    in_file = open('./train_cdc/train_annotations/%s.xml'%(image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()
    width = int(root.find('size').findtext('width'))
    height = int(root.find('size').findtext('height'))

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
    train_file.write(" ")

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
        train_file.write(os.path.abspath(img_dir +'/%s.jpg'%(image_id)))      
        convert_annotation(image_id,train_file)
        if idx != (len(dir_data)-1):
            train_file.write("\n")
        
    train_file.close()
    print("Finish...")


if __name__ == '__main__':
    create_keras_data()

