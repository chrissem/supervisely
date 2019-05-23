# coding: utf-8

import collections
import os
import cv2
import json

import supervisely_lib as sly
from supervisely_lib.io.json import load_json_file
from supervisely_lib.annotation.annotation_transforms import extract_labels_from_mask


class InstanceIdToObjClass(collections.Mapping):
    """
    Mapillary instance ID to Supervisely object class mapping
    to allow us to use common extraction methods for instances masks.
    In Mapillary dataset instance images are in grayscale 16bit format.
    Instance pixels are encoded as: class_index * 256 + instance_index.
    """
    def __init__(self, class_id_mapping: dict):
        self._class_id_mapping = class_id_mapping.copy()

    def __getitem__(self, instance_color_id):
        return self._class_id_mapping[instance_color_id // 256]

    def __len__(self):
        # Length is not wel defined since we are reducing the keys dimensionality.
        raise NotImplementedError

    def __iter__(self):
        # Iterations are not well defined since we are reducing the keys dimensionality.
        raise NotImplementedError()


class ImporterMapillary:

    def __init__(self):
        self.settings = json.load(open(sly.TaskPaths.SETTINGS_PATH))
        self.colors_file = os.path.join(sly.TaskPaths.DATA_DIR, 'config.json')
        self.obj_classes = sly.ObjClassCollection()
        self._read_colors()
        self._read_datasets()

    def _read_datasets(self):
        self.src_datasets = {}
        ds_names = [x for x in os.listdir(sly.TaskPaths.DATA_DIR)
                    if os.path.isdir(os.path.join(sly.TaskPaths.DATA_DIR, x))]
        for ds_name in ds_names:
            imgdir = self._imgs_dir(ds_name)
            sample_names = [os.path.splitext(x)[0]
                            for x in os.listdir(imgdir) if os.path.isfile(os.path.join(imgdir, x))]
            self.src_datasets[ds_name] = sample_names
            sly.logger.info('Found source dataset "{}" with {} sample(s).'.format(ds_name, len(sample_names)))

    def _read_colors(self):
        if os.path.isfile(self.colors_file):
            sly.logger.info('Will try to read segmentation colors from provided file.')
            color_info = load_json_file(self.colors_file)
        else:
            sly.logger.info('Will use default Mapillary color mapping.')
            default_filepath = os.path.join(os.path.dirname(__file__), 'colors.json')
            color_info = load_json_file(default_filepath)

        self._class_id_to_object_class = {
            color_id: sly.ObjClass(name=el['readable'], geometry_type=sly.Bitmap, color=el['color']) for color_id, el
            in enumerate(color_info['labels'])}
        sly.logger.info('Found {} class(es).'.format(len(self._class_id_to_object_class)),
                        extra={
                            'classes': list(obj_class.name for obj_class in self._class_id_to_object_class.values())})
        self._instance_id_to_obj_class = InstanceIdToObjClass(self._class_id_to_object_class)


    @classmethod
    def _read_img_unchanged(cls, img_path):
        return cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # expect uint16

    @staticmethod
    def _imgs_dir(ds_name):
        return os.path.join(sly.TaskPaths.DATA_DIR, ds_name, 'images')

    @staticmethod
    def _segm_dir(ds_name):
        return os.path.join(sly.TaskPaths.DATA_DIR, ds_name, 'labels')

    @staticmethod
    def _inst_dir(ds_name):
        return os.path.join(sly.TaskPaths.DATA_DIR, ds_name, 'instances')

    def _generate_annotation(self, src_img_path, inst_path):
        ann = sly.Annotation.from_img_path(src_img_path)

        if os.path.isfile(inst_path):
            instances_mask = self._read_img_unchanged(inst_path)  # Read uint-16 format
            if ann.img_size != instances_mask.shape:
                instances_mask = cv2.resize(instances_mask, ann.img_size[::-1], interpolation=cv2.INTER_NEAREST)
            labels = extract_labels_from_mask(instances_mask, self._instance_id_to_obj_class)
            ann = ann.add_labels(labels)
        return ann

    def convert(self):
        out_project = sly.Project(os.path.join(sly.TaskPaths.RESULTS_DIR, self.settings['res_names']['project']),
                                  sly.OpenMode.CREATE)

        for ds_name, sample_names in self.src_datasets.items():
            progress = sly.Progress('Dataset: {!r}'.format(ds_name), len(sample_names))
            progress.report_every = 10  # By default progress for 18000 samples report only every 180 - too big.
            ds = out_project.create_dataset(ds_name)

            for name in sample_names:
                img_name = name + '.jpg'
                src_img_path = os.path.join(self._imgs_dir(ds_name), img_name)
                inst_path = os.path.join(self._inst_dir(ds_name), name + '.png')

                try:
                    ann = self._generate_annotation(src_img_path, inst_path)
                    ds.add_item_file(img_name, src_img_path, ann=ann)
                except Exception as e:
                    exc_str = str(e)
                    sly.logger.warn('Input sample skipped due to error: {}'.format(exc_str), exc_info=True, extra={
                        'exc_str': exc_str,
                        'dataset_name': ds_name,
                        'image': src_img_path,
                    })
                progress.iter_done_report()
            sly.logger.info("Dataset '{}' samples processing is done.".format(ds_name), extra={})

        out_meta = sly.ProjectMeta(obj_classes=sly.ObjClassCollection(self._class_id_to_object_class.values()))
        out_project.set_meta(out_meta)
        sly.logger.info("Mapillary samples processing is done.", extra={})


def main():
    importer = ImporterMapillary()
    importer.convert()
    sly.report_import_finished()


if __name__ == '__main__':
    sly.main_wrapper('MAPILLARY_IMPORT', main)
