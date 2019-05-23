# coding: utf-8

import os
import json
import glob
import cv2
import pydicom
import numpy as np

import supervisely_lib as sly


class WrongInputDataStructure(Exception):
    pass


def get_datasets_names_and_paths():
    search_path = os.path.join(sly.TaskPaths.DATA_DIR, '*')
    folders_list = [f for f in glob.glob(search_path) if os.path.isdir(f)]
    files_list = sly.fs.list_files(sly.TaskPaths.DATA_DIR)

    if len(folders_list) > 0 and len(files_list) > 0:
        raise WrongInputDataStructure('Allowed only list if DICOM files OR list of folders which contains DICOM files!'
                                      'See DICOM plugin documentation!')
    if len(files_list) > 0:
        return [('ds', sly.TaskPaths.DATA_DIR)]

    if len(folders_list) > 0:
        return [(os.path.basename(ds_path), ds_path) for ds_path in folders_list]

    raise WrongInputDataStructure('Input data not found!')


def get_tags_from_dicom_object(dicom_obj, requested_tags):
    results = []
    for tag_name in requested_tags:
        tag_value = getattr(dicom_obj, tag_name, None)
        if tag_value is not None:
            tag_meta = sly.TagMeta(tag_name, sly.TagValueType.ANY_STRING)
            tag = sly.Tag(tag_meta, str(tag_value))
            results.append((tag, tag_meta))
    return results


def prepare_dicom_image(image):
    # DICOM image brightness may has range [0...2000] and we need to translate it to [0..255]
    image = (image / (image.mean() / 128.0)).clip(0, 255).astype(np.uint8)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def extract_images_from_dicom(dicom_obj):
    dcm_channels = dicom_obj.pixel_array.shape
    images = []
    if (len(dcm_channels) > 3 and dcm_channels[-1] == 3) or (len(dcm_channels) == 3 and dcm_channels[-1] != 3):
        for i in range(dcm_channels[0]):
            images.append(prepare_dicom_image(dicom_obj.pixel_array[i]))
    else:
        images.append(prepare_dicom_image(dicom_obj.pixel_array))
    return images


def convert():
    settings = json.load(open(sly.TaskPaths.SETTINGS_PATH))
    tag_metas = sly.TagMetaCollection()

    out_project = sly.Project(os.path.join(sly.TaskPaths.RESULTS_DIR, settings['res_names']['project']), sly.OpenMode.CREATE)
    requested_tags = settings.get("options", {}).get("tags", [])

    if len(requested_tags) != len(set(requested_tags)):
        raise ValueError('Duplicate values detected in the list of requested tags: {}'.format(requested_tags))

    skipped_count = 0
    samples_count = 0

    for folder_name, folder_path in get_datasets_names_and_paths():
        dataset = out_project.create_dataset(folder_name)

        # Process all files in current folder
        filenames_in_folder = sly.fs.list_files(folder_path)
        dataset_progress = sly.Progress('Dataset {!r}'.format(folder_name), len(filenames_in_folder))

        for dicom_filename in filenames_in_folder:
            try:
                # Read DICOM file
                dicom_obj = pydicom.dcmread(dicom_filename)

                # Extract tags
                tags_and_metas = get_tags_from_dicom_object(dicom_obj, requested_tags)

                # Extract images (DICOM file may contain few images)
                base_name = os.path.splitext(os.path.basename(dicom_filename))[0]
                images = extract_images_from_dicom(dicom_obj)

                for image_index, image in enumerate(images):
                    sample_name = base_name
                    if len(images) > 1:
                        sample_name = base_name + '__{:04d}'.format(image_index)

                    samples_count += 1
                    ann = sly.Annotation(img_size=image.shape[:2])

                    # Save tags
                    for tag, tag_meta in tags_and_metas:
                        ann = ann.add_tag(tag)
                        if tag_meta not in tag_metas:
                            tag_metas = tag_metas.add(tag_meta)

                    # Save annotations
                    dataset.add_item_np(sample_name + '.png', image, ann=ann)

            except Exception as e:
                exc_str = str(e)
                sly.logger.warn('Input sample skipped due to error: {}'.format(exc_str), exc_info=True, extra={
                    'exc_str': exc_str,
                    'dataset_name': folder_name,
                    'image_name': dicom_filename,
                })
                skipped_count += 1

            dataset_progress.iter_done_report()

    sly.logger.info('Processed.', extra={'samples': samples_count, 'skipped': skipped_count})

    out_meta = sly.ProjectMeta(tag_metas=tag_metas)
    out_project.set_meta(out_meta)


def main():
    convert()
    sly.report_import_finished()


if __name__ == '__main__':
    sly.main_wrapper('IMAGES_ONLY_IMPORT', main)