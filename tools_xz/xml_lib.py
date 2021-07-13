import sys
import os

import xml.etree.ElementTree as Et
from xml.etree.ElementTree import Element, ElementTree
from tqdm import tqdm
import numpy as np
import cv2, json 

'''
meta data:
{
'img_filename':str,
'objs_list':[{'name':name, 'xyxy':(x,y,x,y),'conf':float..},....],
'img_whc':(int w,h,c),
'data_base':str, ##
'obj_num':int, ##
'img_dir':str, ##
}
'''
'''
labelme json data:
{
  "version": "4.5.6",
  "flags": {},
  "shapes": [
    {
      "label": "face",
      "points": [
        [
          900.695652173913, 
          348.0579710144927 
        ],
        [
          1267.3623188405795,
          716.1739130434783
        ]
      ],
      "group_id": null,
      "shape_type": "rectangle",
      "flags": {}
    },
    
  ],
  "imagePath": "001_20210629_025530_329.jpg",
  "imageData": null,
  "imageHeight": 1080,
  "imageWidth": 1920
}
'''


class Xml:

    def __init__(self):
        self.info = 'xml class'

    def xml_indent(self, elem, level=0):
        i = "\n" + level * "\t"
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "\t"
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.xml_indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def _generateXmlElement(self, annotation_dic):
        img_filename = annotation_dic['img_filename']
        img_w, img_h, img_c = annotation_dic['img_whc']
        objs_list = annotation_dic['objs_list']

        filename = os.path.split(img_filename)[-1]
        xml_annotation = Element("annotation")

        xml_folder = Element("folder")
        xml_folder.text = ' '
        xml_filename = Element("filename")
        xml_filename.text = filename
        xml_path = Element("path")
        xml_path.text = filename
        xml_source = Element("source")
        xml_database = Element("database")
        xml_database.text = 'Unknown'
        xml_source.append(xml_database)
        xml_annotation.append(xml_folder)
        xml_annotation.append(xml_filename)
        xml_annotation.append(xml_path)
        xml_annotation.append(xml_source)

        xml_size = Element("size")
        xml_width = Element("width")
        xml_width.text = str(img_w)
        xml_size.append(xml_width)

        xml_height = Element("height")
        xml_height.text = str(img_h)
        xml_size.append(xml_height)

        xml_depth = Element("depth")
        xml_depth.text = str(img_c)
        xml_size.append(xml_depth)

        xml_annotation.append(xml_size)

        xml_segmented = Element("segmented")
        xml_segmented.text = "0"

        xml_annotation.append(xml_segmented)

        if len(objs_list) < 1:
            print('Error number of Object less than 1')

        for i, item in enumerate(objs_list):
            item_name = item['name']
            x1, y1, x2, y2 = item['xyxy']

            xml_object = Element("object")
            obj_name = Element("name")
            obj_name.text = item_name
            xml_object.append(obj_name)

            obj_pose = Element("pose")
            obj_pose.text = "Unspecified"
            xml_object.append(obj_pose)

            obj_truncated = Element("truncated")
            obj_truncated.text = "0"
            xml_object.append(obj_truncated)

            obj_difficult = Element("difficult")
            obj_difficult.text = "0"
            xml_object.append(obj_difficult)

            xml_bndbox = Element("bndbox")

            obj_xmin = Element("xmin")
            obj_xmin.text = str(int(x1))
            xml_bndbox.append(obj_xmin)

            obj_ymin = Element("ymin")
            obj_ymin.text = str(int(y1))
            xml_bndbox.append(obj_ymin)

            obj_xmax = Element("xmax")
            obj_xmax.text = str(int(x2))
            xml_bndbox.append(obj_xmax)

            obj_ymax = Element("ymax")
            obj_ymax.text = str(int(y2))
            xml_bndbox.append(obj_ymax)
            xml_object.append(xml_bndbox)

            xml_annotation.append(xml_object)

        self.xml_indent(xml_annotation)

        return xml_annotation

    def _generateLabelMeJson(self, annotation_dic):
        data = []
        img_whc = annotation_dic['img_whc']
        img_filename = annotation_dic['img_filename']
        img_filename = os.path.split(img_filename)[-1]
        objs_list = annotation_dic['objs_list']
        res = {
            "version": "4.5.6",
            "flags": {},
            "shapes": data,
            "imagePath": img_filename,
            "imageData": None,
            "imageHeight": int(img_whc[1]),
            "imageWidth": int(img_whc[0])
            } 
        for obj in objs_list:
            item = []
            label = obj['name']
            x1,y1,x2,y2 = [float(i) for i in obj['xyxy']]
            item = {
                    "label": label,
                    "points": [[x1, y1],[x2,y2]],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                    }
            data.append(item)
        return res 


    def resultToAnnotationDic(self, xywhcs, class_name, img_filename, img_whc, is_xyxy=False):
        annotation_dic = {}
        annotation_dic['img_filename'] = img_filename
        annotation_dic['img_whc'] = img_whc
        objs_list = []
        objs_num = 0
        for xyhwc in xywhcs:
            # [{'name':name, 'xyxy':(x,y,x,y)},....]
            if not is_xyxy:
                o_xmin, o_ymin, w, h, conf = xyhwc
                o_xmax, o_ymax = o_xmin + w, o_ymin + h
            else:
                o_xmin, o_ymin, o_xmax, o_ymax, conf = xyhwc
            o_name = class_name
            objs_list.append({'name': o_name, 'xyxy': (o_xmin, o_ymin, o_xmax, o_ymax)})
            objs_num += 1
        annotation_dic['objs_list'] = objs_list
        annotation_dic['objs_num'] = objs_num
        return annotation_dic

    def resultToXml(self, xywhcs_list, class_name_list, img_filename, img_whc, score_thr):
        for i, xywhcs in enumerate(xywhcs_list):
            class_name = class_name_list[i]
            xywhcs = xywhcs[xywhcs[:, -1] > score_thr]

            annotation_dic = self.resultToAnnotationDic(xywhcs, class_name, img_filename, img_whc, is_xyxy=False)
            xml_element = self._generateXmlElement(annotation_dic)
            end_fix = img_filename.split('.')[-1]
            xml_filename = img_filename.replace(end_fix, 'xml')
            try:
                Et.ElementTree(xml_element).write(xml_filename)
                # print('save xml file:', xml_filename)
            except Exception as e:
                print(e)
            return xml_filename

    def mmdetResultToXml(self, mmdet_res, class_name_list, img_filename, xml_save_pth, img_whc, score_thr, is_xyxy=False):
        ann = None
        for i, xywhcs in enumerate(mmdet_res):
            if len(xywhcs) == 0:
                continue
            class_name = class_name_list[i]
            xywhcs = xywhcs[xywhcs[:, -1] > score_thr]

            annotation_dic = self.resultToAnnotationDic(xywhcs, class_name, img_filename, img_whc, is_xyxy=is_xyxy)
            if ann is None:
                ann = annotation_dic
            else:
                ann, _ = self.mergeAnn(ann, annotation_dic)

        xml_element = self._generateXmlElement(ann)

        if xml_save_pth is None:
            # end_fix = img_filename.split('.')[-1]
            # xml_filename = img_filename.replace(end_fix, 'xml')
            raise FileNotFoundError
        else:
            xml_filename = xml_save_pth
        try:
            Et.ElementTree(xml_element).write(xml_filename)
            print('save xml file:', xml_filename)
        except Exception as e:
            print(e)
        return xml_filename

    def mmdetResultToLabelMeJson(self, mmdet_res, class_name_list, img_filename, json_save_pth, img_whc, score_thr, is_xyxy=False):
        '''
        '''
        ann = None
        for i, xywhcs in enumerate(mmdet_res):
            if len(xywhcs) == 0:
                continue
            class_name = class_name_list[i]
            xywhcs = xywhcs[xywhcs[:, -1] > score_thr]

            annotation_dic = self.resultToAnnotationDic(xywhcs, class_name, img_filename, img_whc, is_xyxy=is_xyxy)
            if ann is None:
                ann = annotation_dic
            else:
                ann, _ = self.mergeAnn(ann, annotation_dic)

        json_data = self._generateLabelMeJson(ann)

        if json_save_pth is None:
            raise FileNotFoundError

        try:
            with open(json_save_pth, 'w') as f:
                json.dump(json_data,f, indent=4)
            print('save json file:', json_save_pth)
        except Exception as e:
            print(e)
        return json_save_pth 

    def writeXyxyToXml(self, xyxycs_class, name_list, img_name, img_whc, conf_th=0.0):
        annotation_dic = {}
        annotation_dic['img_filename'] = img_name
        annotation_dic['img_whc'] = img_whc
        objs_list = []
        objs_num = 0
        for i, xyxyc_1class in enumerate(xyxycs_class):
            if len(xyxyc_1class) == 0:
                continue
            o_name = name_list[i]
            for xyxyc in xyxyc_1class:
                o_xmin, o_ymin, o_xmax, o_ymax, conf = xyxyc
                if conf < conf_th:
                    continue
                objs_list.append({'name': o_name, 'xyxy': (o_xmin, o_ymin, o_xmax, o_ymax)})
                objs_num += 1
        annotation_dic['objs_list'] = objs_list
        annotation_dic['objs_num'] = objs_num

        end_fix = os.path.splitext(img_name)[-1]
        xml_filename = img_name.replace(end_fix, '.xml')
        self.writeXml(annotation_dic, save_pth=xml_filename)

    def writeXml(self, annotation_dic, save_pth=None):
        xml_element = self._generateXmlElement(annotation_dic)
        img_filename = annotation_dic['img_filename']
        end_fix = os.path.splitext(img_filename)[-1]
        xml_filename = img_filename.replace(end_fix, '.xml')
        if save_pth is not None:
            xml_filename = save_pth
        try:
            Et.ElementTree(xml_element).write(xml_filename)
        except Exception as e:
            print(e)
        return xml_filename

    def readXml(self, xml_pth):
        xml = open(xml_pth, "r")
        try:
            tree = Et.parse(xml)
        except Exception as e:
            print('Error read:', xml_pth)
            return None
        root = tree.getroot()

        filename = root.find('filename').text
        try:
            database = root.find('source').find('database').text
            path = root.find('path').text
        except Exception as e:
            database = 'Unknown'
            path = ' '

        xml_size = root.find("size")
        img_w = int(xml_size.find("width").text)
        img_h = int(xml_size.find("height").text)
        img_c = int(xml_size.find("depth").text)
        size_whc = (img_w, img_h, img_c)

        objects = root.findall("object")
        if len(objects) == 0:
            print('\nno obj in ', filename)
            return None

        objs_num = 0
        objs_list = []
        for _object in objects:
            # [{'name':name, 'xyxy':(x,y,x,y)},....]
            o_name = _object.find("name").text
            xml_bndbox = _object.find("bndbox")
            o_xmin = int(xml_bndbox.find("xmin").text)
            o_ymin = int(xml_bndbox.find("ymin").text)
            o_xmax = int(xml_bndbox.find("xmax").text)
            o_ymax = int(xml_bndbox.find("ymax").text)
            temp = {'name': o_name, 'xyxy': (o_xmin, o_ymin, o_xmax, o_ymax)}
            objs_list.append(temp)
            objs_num += 1

        annotation_dic = {
            "img_whc": size_whc,
            "objs_list": objs_list,
            "img_filename": filename,
            "objs_num": objs_num,
            "xml_pth": xml_pth
        }

        return annotation_dic

    def mergeAnn(self, anno1, anno2):
        match = True
        try:
            assert anno1['img_whc'] == anno2['img_whc'], 'Error img size is different'
        except Exception as e:
            print('Error:', anno1['img_whc'], anno2['img_whc'])
            print('Error:miss match:', anno1['img_filename'])
            print('Error:miss match:', anno2['img_filename'])
            match = False
            # return None

        anno1['objs_list'] += anno2['objs_list']
        anno1['objs_num'] += anno2['objs_num']
        return anno1, match


if __name__ == '__main__':
    xml_writer = Xml()
    img_file = './test.jpg'
    import cv2

    img = cv2.imread(img_file)
    # img_whc = (img.shape[1], img.shape[0], 3)
    img_whc = (224,224,3)

    # ann1 = xml_writer.readXml('/workspace/code/detection/convert2Yolo/images/voc_image.xml')
    # ann2 = xml_writer.readXml('/workspace/code/detection/convert2Yolo/images/voc_image1.xml')
    # ann3 = xml_writer.mergeAnn(ann1, ann2)
    # xml_writer.writeXml(ann3, './')

    xml_writer.resultToXml(np.array([[[511, 322, 50, 50, 0.9], [112, 224, 50, 50, 0.3]]]), ['yun'], img_file, img_whc,
                           0.2)

    '/home/asus/opt/user/xz_2020/code/code/deploy/demo-patterns-detect/results'
