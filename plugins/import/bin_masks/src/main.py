# coding: utf-8

import json
import cv2
import numpy as np
import os


from os.path import join
from PIL import Image
from typing import List, Dict

import supervisely_lib as sly
from supervisely_lib.io.json import load_json_file


IMAGE_DIR_NAME = 'img'
ANNOTATION_DIR_NAME = 'ann'
DATASET_NAME = 'ds'

OPTIONS = 'options'  # TODO: take out common keys for all imports
CLASSES_MAPPING_KEY = 'classes_mapping'
MATCH_ALL = '__all__'
DEFAULT_CLASSES_MAPPING = {'untitled': MATCH_ALL}


def read_image_pillow(image_fp: str) -> Image:
    image = np.array(Image.open(image_fp))
    return image


def create_obj_class_collection(classes_mapping: Dict) -> sly.ObjClassCollection:
    cls_list = [sly.ObjClass(cls_name, sly.Bitmap) for cls_name in classes_mapping.keys()]
    return sly.ObjClassCollection(cls_list)


def read_mask_labels(mask_path: str, classes_mapping: Dict, obj_classes: sly.ObjClassCollection) -> List[sly.Label]:
    mask = cv2.imread(mask_path)[:, :, 0]
    labels_list = []
    for cls_name, color in classes_mapping.items():
        if color == MATCH_ALL:
            bool_mask = mask > 0
        elif isinstance(color, int):
            bool_mask = mask == color
        elif isinstance(color, list):
            bool_mask = np.isin(mask, color)
        else:
            raise ValueError('Wrong color format. It must be integer, list of integers or special key string "__all__".')

        if bool_mask.sum() == 0:
            continue

        bitmap = sly.Bitmap(data=bool_mask)
        obj_class = obj_classes.get(cls_name)
        labels_list.append(sly.Label(geometry=bitmap, obj_class=obj_class))
    return labels_list


def convert():
    img_dir = join(sly.TaskPaths.DATA_DIR, IMAGE_DIR_NAME)
    ann_dir = join(sly.TaskPaths.DATA_DIR, ANNOTATION_DIR_NAME)
    tasks_settings = load_json_file(sly.TaskPaths.TASK_CONFIG_PATH)

    classes_mapping = DEFAULT_CLASSES_MAPPING
    if CLASSES_MAPPING_KEY in tasks_settings[OPTIONS]:
        classes_mapping = tasks_settings[OPTIONS][CLASSES_MAPPING_KEY]
    else:
        sly.logger.warn('Classes mapping not found. Set to default: {}'.format(str(DEFAULT_CLASSES_MAPPING)))

    pr = sly.Project(os.path.join(sly.TaskPaths.RESULTS_DIR, tasks_settings['res_names']['project']),
                     sly.OpenMode.CREATE)
    obj_class_collection = create_obj_class_collection(classes_mapping)
    pr_meta = sly.ProjectMeta(obj_classes=obj_class_collection)
    pr.set_meta(pr_meta)
    ds = pr.create_dataset(DATASET_NAME)

    images_pathes = sly.fs.list_files(img_dir)
    masks_pathes = sly.fs.list_files(ann_dir)
    masks_map = {sly.fs.get_file_name(mask_p): mask_p for mask_p in masks_pathes}

    progress = sly.Progress('Dataset: {!r}'.format(DATASET_NAME), len(images_pathes))
    for img_fp in images_pathes:
        full_img_fp = join(img_dir, img_fp)
        try:
            image = read_image_pillow(full_img_fp)
            image_name = os.path.basename(full_img_fp)
            sample_name = sly.fs.get_file_name(full_img_fp)

            ann = sly.Annotation(image.shape[:2])
            mask_name = masks_map.pop(sample_name, None)
            if mask_name is None:
                sly.logger.warning('Mask for image {} doesn\'t exist.'.format(sample_name))
            else:
                full_mask_fp = join(ann_dir, mask_name)
                labels = read_mask_labels(full_mask_fp, classes_mapping, obj_class_collection)
                ann = ann.add_labels(labels)

            ds.add_item_np(image_name, image, ann=ann)
        except Exception as e:
            exc_str = str(e)
            sly.logger.warn('Input sample skipped due to error: {}'.format(exc_str), exc_info=True, extra={
                'exc_str': exc_str,
                'image': full_img_fp
            })
        progress.iter_done_report()

    if len(masks_map) > 0:
        masks_list = list(masks_map.values())
        sly.logger.warning('Images for masks doesn\'t exist. Masks: {}'.format(masks_list))



def main():
    convert()
    sly.report_import_finished()


if __name__ == '__main__':
    sly.main_wrapper('BINARY_MASKS_IMPORT', main)
