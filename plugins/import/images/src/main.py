# coding: utf-8

import os
import supervisely_lib as sly
from supervisely_lib.io.json import load_json_file


def find_input_datasets():
    def flat_images_found(dir_path):
        img_fnames = sly.fs.list_files(dir_path, filter_fn=sly.image.has_valid_ext)
        res = len(img_fnames) > 0
        return res

    def collect_ds_names(pr_path):
        subdirs = sly.fs.get_subdirs(pr_path)
        subd_with_paths = [(x, os.path.join(pr_path, x)) for x in subdirs]
        res = list(filter(lambda x: flat_images_found(x[1]), subd_with_paths))
        return res

    if flat_images_found(sly.TaskPaths.DATA_DIR):
        sly.logger.info('Input structure: flat set of images.')
        return [('ds', sly.TaskPaths.DATA_DIR), ]

    in_datasets = collect_ds_names(sly.TaskPaths.DATA_DIR)
    if len(in_datasets) > 0:
        sly.logger.info('Input structure: set of dirs (datasets) with images.', extra={'ds_cnt': len(in_datasets)})
        return in_datasets
    else:
        top_subdirs = sly.fs.get_subdirs(sly.TaskPaths.DATA_DIR)
        if len(top_subdirs) == 1:
            new_in_dir = os.path.join(sly.TaskPaths.DATA_DIR, top_subdirs[0])
            in_datasets = collect_ds_names(new_in_dir)
            if len(in_datasets) > 0:
                sly.logger.info('Input structure: dir with set of dirs (datasets) with images.',
                            extra={'ds_cnt': len(in_datasets)})
                return in_datasets
            else:
                raise RuntimeError('Input directory is empty')
        elif len(top_subdirs) == 0:
            raise RuntimeError('Input directory is empty')
        else:
            raise RuntimeError('Too many subfolders in the input directory')


def convert():
    task_settings = load_json_file(sly.TaskPaths.TASK_CONFIG_PATH)
    in_datasets = find_input_datasets()

    pr = sly.Project(os.path.join(sly.TaskPaths.RESULTS_DIR, task_settings['res_names']['project']),
                     sly.OpenMode.CREATE)
    for ds_name, ds_path in in_datasets:
        img_paths = sly.fs.list_files(ds_path, filter_fn=sly.image.has_valid_ext)
        sly.logger.info(
            'Found {} files with supported image extensions in Dataset {!r}.'.format(len(img_paths), ds_name))
        ds = pr.create_dataset(ds_name)
        progress = sly.Progress('Dataset: {!r}'.format(ds_name), len(img_paths))
        for img_path in img_paths:
            try:
                item_name = os.path.basename(img_path)
                if ds.item_exists(item_name):
                    item_name_noext, item_ext = os.path.splitext(item_name)
                    item_name = item_name_noext + '_' + sly.rand_str(5) + item_ext
                ds.add_item_file(item_name, img_path)
            except Exception as e:
                exc_str = str(e)
                sly.logger.warn('Input sample skipped due to error: {}'.format(exc_str), exc_info=True, extra={
                    'exc_str': exc_str,
                    'dataset_name': ds_name,
                    'image_name': img_path,
                })
            progress.iter_done_report()


def main():
    convert()
    sly.report_import_finished()


if __name__ == '__main__':
    sly.main_wrapper('IMPORT_IMAGES', main)
