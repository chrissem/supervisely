# coding: utf-8

import os
import cv2
import numpy as np
import supervisely_lib as sly
from supervisely_lib.io.json import load_json_file
from supervisely_lib.annotation.annotation_transforms import extract_labels_from_mask


CLASSES_COLORS = {
    'background': (0, 0, 143),
    'hair': (0, 32, 255),
    'face': (0, 191, 255),
    'upperclothes': (96, 255, 159),
    'lowerclothes': (255, 255, 0),
    'arms': (255, 80, 0),
    'shoes': (143, 0, 0),
    'body': (175, 0, 0)
}


def code_color(r, g, b):  # Encode 3-component color to 1-component (unique)
    return r * 1000000 + g * 1000 + b


class PedestrianConverter:
    def __init__(self, class_title_to_color):
        self.class_title_to_color = class_title_to_color
        self.color_id_to_class_title = {code_color(*color): class_title
                                        for class_title, color in class_title_to_color.items()}

        self.id_to_obj_class = {
            color_id: sly.ObjClass(
                name=class_name,
                geometry_type=sly.Bitmap,
                color=self.class_title_to_color[class_name])
            for color_id, class_name in self.color_id_to_class_title.items()}

        self.settings = load_json_file(sly.TaskPaths.TASK_CONFIG_PATH)
        self.src_datasets = self.read_datasets_from_path()

    @staticmethod
    def read_datasets_from_path():
        root_path = os.path.join(sly.TaskPaths.DATA_DIR, 'pedestrian_parsing_dataset/data')
        if not os.path.isdir(root_path):
            raise RuntimeError("There is no directory '{}', but it is necessary".format(root_path))

        datasets = {}
        filter_fn = lambda name: '_m.' not in name  # no-mask pattern e.g: 01_m.png
        for dir_name in sly.fs.get_subdirs(root_path):
            datasets[dir_name] = sly.fs.list_files(os.path.join(root_path, dir_name), filter_fn=filter_fn)
        return datasets


    def get_ann(self, img_path, inst_path):
        image = sly.image.read(img_path)
        ann = sly.Annotation.from_img_path(img_path)

        instances_mask = sly.image.read(inst_path)
        instances_mask = cv2.resize(instances_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        instances_mask = instances_mask.astype(np.uint32)
        instances_mask = code_color(instances_mask[:, :, 0], instances_mask[:, :, 1], instances_mask[:, :, 2])

        labels = extract_labels_from_mask(instances_mask, self.id_to_obj_class)
        return ann.add_labels(labels)

    @staticmethod
    def get_ann_path(img_path):
        return os.path.splitext(img_path)[0] + '_m.png'

    def convert(self):
        out_project = sly.Project(os.path.join(sly.TaskPaths.RESULTS_DIR, self.settings['res_names']['project']),
                                  sly.OpenMode.CREATE)

        progress = sly.Progress('Dataset:', len(self.src_datasets))
        for ds_name, samples_paths in self.src_datasets.items():
            ds = out_project.create_dataset(ds_name)

            for src_img_path in samples_paths:
                try:
                    ann_path = self.get_ann_path(src_img_path)
                    if all((os.path.isfile(x) for x in [src_img_path, ann_path])):
                        ann = self.get_ann(src_img_path, ann_path)
                        ds.add_item_file(os.path.basename(src_img_path), src_img_path, ann=ann)
                except Exception as e:
                    exc_str = str(e)
                    sly.logger.warn('Input sample skipped due to error: {}'.format(exc_str), exc_info=True, extra={
                        'exc_str': exc_str,
                        'dataset_name': ds_name,
                        'image_name': src_img_path,
                    })
            progress.iter_done_report()

        out_meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(self.id_to_obj_class.values()))
        out_project.set_meta(out_meta)


def main():
    converter = PedestrianConverter(CLASSES_COLORS)
    converter.convert()
    sly.report_import_finished()


if __name__ == '__main__':
    sly.main_wrapper('pedestrian', main)
