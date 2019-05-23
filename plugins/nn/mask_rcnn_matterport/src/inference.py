# coding: utf-8

import os
import numpy as np

import supervisely_lib as sly
from supervisely_lib.annotation.annotation import Annotation
from supervisely_lib.sly_logger import logger
from supervisely_lib.task.paths import TaskPaths
from supervisely_lib.nn.hosted.constants import SETTINGS
from supervisely_lib.nn.hosted.inference_single_image import SingleImageInferenceBase, GPU_DEVICE
from supervisely_lib.nn.hosted.inference_batch import BatchInferenceApplier
from supervisely_lib.nn.hosted.inference_modes import InfModeFullImage, CONFIDENCE
from supervisely_lib.nn.config import JsonConfigValidator

import inference_lib
import custom_config


class MaskRCNNSingleImageApplier(SingleImageInferenceBase):

    @staticmethod
    def get_default_config():
        return {
            GPU_DEVICE: 0,
            'confidence_tag_name': CONFIDENCE
        }

    @property
    def train_classes_key(self):
        return custom_config.train_classes_key()

    @property
    def class_title_to_idx_key(self):
        return custom_config.class_to_idx_config_key()

    def _load_train_config(self):
        self.confidence_tag_meta = sly.TagMeta(self._config['confidence_tag_name'], sly.TagValueType.ANY_NUMBER)
        super()._load_train_config()
        src_size = self.train_config[SETTINGS]['input_size']
        self.input_size_limits = (src_size['min_dim'], src_size['max_dim'])
        self.idx_to_class_title = {v: k for k, v in self.class_title_to_idx.items()}

    def _validate_model_config(self, config):
        JsonConfigValidator().validate_inference_cfg(config)

    def _model_out_tags(self):
        tag_meta_dict = sly.TagMetaCollection()
        return tag_meta_dict.add(self.confidence_tag_meta)

    def _construct_and_fill_model(self):
        super()._construct_and_fill_model()
        self.device_ids = sly.env.remap_gpu_devices([self._config[GPU_DEVICE]])
        self.graph, self.model = inference_lib.construct_and_fill_model(self.class_title_to_idx, self.input_size_limits,
                                                                        TaskPaths.MODEL_DIR)
        logger.info('Weights are loaded.')

        logger.info('Warming up the model with a dummy image.')
        with self.graph.as_default():
            self.model.detect([np.zeros([256, 256, 3], dtype=np.uint8)], verbose=0)
        logger.info('Model warmup finished.')

    def inference(self, image, ann):
        res_labels = inference_lib.infer_on_image(image, self.graph, self.model, self.idx_to_class_title,
                                                  self.model_out_meta, self.confidence_tag_meta)
        return Annotation(ann.img_size, labels=res_labels)


def main():
    single_image_applier = MaskRCNNSingleImageApplier()
    default_inference_mode_config = InfModeFullImage.make_default_config(model_result_suffix='_maskrcnn')
    dataset_applier = BatchInferenceApplier(single_image_inference=single_image_applier,
                                            default_inference_mode_config=default_inference_mode_config,
                                            config_validator=JsonConfigValidator())
    dataset_applier.run_inference()


if __name__ == '__main__':
    if os.getenv('DEBUG_LOG_TO_FILE', None):
        sly.add_default_logging_into_file(logger, sly.TaskPaths.DEBUG_DIR)
    sly.main_wrapper('MASK_RCNN_MATTERPORT_INFERENCE', main)
