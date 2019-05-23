# coding: utf-8

import os
import glob
import json

import supervisely_lib as sly


IMAGE_EXT = '.png'


class AnnotationConvertionException(Exception):
    pass


class ImporterCityscapes:
    def __init__(self):
        self.settings = json.load(open(sly.TaskPaths.SETTINGS_PATH))
        self.obj_classes = sly.ObjClassCollection()
        self.tag_metas = sly.TagMetaCollection()

    @classmethod
    def json_path_to_image_path(cls, json_path):
        img_path = json_path.replace('/gtFine/', '/leftImg8bit/')
        img_path = img_path.replace('_gtFine_polygons.json', '_leftImg8bit' + IMAGE_EXT)
        return img_path

    @staticmethod
    def convert_points(simple_points):
        # TODO: Maybe use row_col_list_to_points here?
        return [sly.PointLocation(int(p[1]), int(p[0])) for p in simple_points]

    def _load_cityscapes_annotation(self, orig_img_path, orig_ann_path) -> sly.Annotation:
        json_data = json.load(open(orig_ann_path))
        ann = sly.Annotation.from_img_path(orig_img_path)

        for obj in json_data['objects']:
            class_name = obj['label']
            if class_name == 'out of roi':
                polygon = obj['polygon'][:5]
                interiors = [obj['polygon'][5:]]
            else:
                polygon = obj['polygon']
                interiors = []

            interiors = [self.convert_points(interior) for interior in interiors]
            polygon = sly.Polygon(self.convert_points(polygon), interiors)
            obj_class = sly.ObjClass(name=class_name, geometry_type=sly.Polygon, color=sly.color.random_rgb())
            ann = ann.add_label(sly.Label(polygon, obj_class))
            if not self.obj_classes.has_key(class_name):
                self.obj_classes = self.obj_classes.add(obj_class)
        return ann

    def _generate_sample_annotation(self, orig_img_path, orig_ann_path, train_val_tag):
        try:
            tag_meta = sly.TagMeta(train_val_tag, sly.TagValueType.NONE)
            if not self.tag_metas.has_key(tag_meta.name):
                self.tag_metas = self.tag_metas.add(tag_meta)
            tag = sly.Tag(tag_meta)
            ann = self._load_cityscapes_annotation(orig_img_path, orig_ann_path)
            ann = ann.add_tag(tag)
            return ann
        except Exception:
            raise AnnotationConvertionException()  # ok, may continue work with another sample

    def convert(self):
        search_fine = os.path.join(sly.TaskPaths.DATA_DIR, "gtFine", "*", "*", "*_gt*_polygons.json")
        files_fine = glob.glob(search_fine)
        files_fine.sort()

        search_imgs = os.path.join(sly.TaskPaths.DATA_DIR, "leftImg8bit", "*", "*", "*_leftImg8bit" + IMAGE_EXT)
        files_imgs = glob.glob(search_imgs)
        files_imgs.sort()

        out_project = sly.Project(os.path.join(sly.TaskPaths.RESULTS_DIR, self.settings['res_names']['project']), sly.OpenMode.CREATE)

        samples_count = len(files_fine)
        progress = sly.Progress('Project: {!r}'.format(out_project.name), samples_count)

        ok_count = 0
        for orig_ann_path in files_fine:
            parent_dir, json_filename = os.path.split(os.path.abspath(orig_ann_path))

            dataset_name = os.path.basename(parent_dir)
            ds = out_project.datasets.get(dataset_name)
            if ds is None:
                ds = out_project.create_dataset(dataset_name)

            sample_name = json_filename.replace('_gtFine_polygons.json', IMAGE_EXT)
            orig_img_path = self.json_path_to_image_path(orig_ann_path)

            tag_path = os.path.split(parent_dir)[0]
            train_val_tag = os.path.basename(tag_path)  # e.g. train, val, test

            try:
                ann = self._generate_sample_annotation(orig_img_path, orig_ann_path, train_val_tag)

                if all(os.path.isfile(x) for x in (orig_img_path, orig_ann_path)):
                    ds.add_item_file(sample_name, orig_img_path, ann=ann)
                else:
                    sly.logger.warn('Skipped sample without a complete set of files: {}'.format(sample_name),
                                    exc_info=False, extra={'sample_name': sample_name,
                                                           'image_path': orig_img_path,
                                                           'annotation_path': orig_ann_path})

            except AnnotationConvertionException:
                sly.logger.warn('Error occurred while processing input sample annotation.',
                                exc_info=True, extra={'sample_name': sample_name})
            except Exception:
                sly.logger.error('Error occurred while processing input sample.',
                                 exc_info=False, extra={'sample_name': sample_name})
                raise
            else:
                ok_count += 1
            progress.iter_done_report()

        stat_dct = {'samples': samples_count, 'src_ann_cnt': len(files_fine), 'src_img_cnt': len(files_imgs)}

        sly.logger.info('Found img/ann pairs.', extra=stat_dct)
        if stat_dct['samples'] < stat_dct['src_ann_cnt']:
            sly.logger.warn('Found source annotations without corresponding images.', extra=stat_dct)

        sly.logger.info('Processed.', extra={'samples': samples_count, 'ok_cnt': ok_count})

        out_meta = sly.ProjectMeta(obj_classes=self.obj_classes, tag_metas=self.tag_metas)
        out_project.set_meta(out_meta)
        sly.logger.info('Found classes.', extra={'cnt': len(self.obj_classes),
                                                 'classes': sorted([obj_class.name for obj_class in self.obj_classes])})
        sly.logger.info('Created tags.', extra={'cnt': len(out_meta.tag_metas),
                                                'tags': sorted([tag_meta.name for tag_meta in out_meta.tag_metas])})


def main():
    importer = ImporterCityscapes()
    importer.convert()
    sly.report_import_finished()


if __name__ == '__main__':
    sly.main_wrapper('CITYSCAPES_IMPORT', main)
