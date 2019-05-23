# coding: utf-8

import os
import numpy as np

import supervisely_lib as sly
from supervisely_lib import sly_logger
from supervisely_lib.annotation.annotation import Annotation
from supervisely_lib.annotation.obj_class_collection import ObjClassCollection
from supervisely_lib.annotation.tag import Tag
from supervisely_lib.annotation.tag_collection import TagCollection
from supervisely_lib.annotation.tag_meta import TagMeta, TagValueType
from supervisely_lib.annotation.tag_meta_collection import TagMetaCollection
from supervisely_lib.project.project_meta import ProjectMeta
from supervisely_lib.sly_logger import logger
from supervisely_lib.task.paths import TaskPaths
from supervisely_lib.nn.hosted.inference_single_image import SingleImageInferenceBase, GPU_DEVICE
from supervisely_lib.nn.hosted.inference_batch import BatchInferenceApplier
from supervisely_lib.nn.hosted.inference_modes import InfModeFullImage
from supervisely_lib.nn.config import JsonConfigValidator
from supervisely_lib.nn.pytorch.weights import WeightsRW

import config as config_lib

from common import infer_on_img, determine_resnet_model_configuration
from model_utils import create_model


class ResnetSingleImageApplier(SingleImageInferenceBase):
    @property
    def classification_tags_key(self):
        return config_lib.classification_tags_key()

    @property
    def classification_tags_to_idx_key(self):
        return config_lib.classification_tags_to_idx_key()

    @property
    def train_classes_key(self):
        return config_lib.train_classes_key()

    @property
    def class_title_to_idx_key(self):
        return config_lib.class_to_idx_config_key()

    def _model_out_tags(self):
        temp_collection = TagMetaCollection.from_json(self.train_config[self.classification_tags_key])
        res_collection = TagMetaCollection([TagMeta(x.name, TagValueType.ANY_NUMBER) for x in temp_collection])
        return res_collection

    def _load_train_config(self):  # @TODO: partly copypasted from SingleImageInferenceBase
        self._load_raw_model_config_json()

        self.classification_tags = self._model_out_tags()
        logger.info('Read model out tags', extra={'tags': self.classification_tags.to_json()})
        self.classification_tags_to_idx = self.train_config[self.classification_tags_to_idx_key]
        logger.info('Read model internal tags mapping', extra={'tags_mapping': self.classification_tags_to_idx})

        self._model_out_meta = ProjectMeta(obj_classes=ObjClassCollection(), tag_metas=self.classification_tags)

        self.idx_to_classification_tags = {v: k for k, v in self.classification_tags_to_idx.items()}
        self._determine_model_input_size()

    def _validate_model_config(self, config):
        JsonConfigValidator().validate_inference_cfg(config)

    @staticmethod
    def get_default_config():
        return {
            GPU_DEVICE: 0
        }

    def _construct_and_fill_model(self):
        super()._construct_and_fill_model()
        device_ids = sly.env.remap_gpu_devices([self._config[GPU_DEVICE]])
        num_layers = determine_resnet_model_configuration(TaskPaths.MODEL_CONFIG_PATH)
        self.model = create_model(num_layers=num_layers, n_cls=(max(self.classification_tags_to_idx.values()) + 1),
                                  device_ids=device_ids)

        self.model = WeightsRW(TaskPaths.MODEL_DIR).load_strictly(self.model)
        self.model.eval()
        logger.info('Weights are loaded.')

    def inference(self, img, ann):
        output = infer_on_img(img, self.input_size, self.model)
        tag_id = np.argmax(output)
        score = output[tag_id]
        tag_name = self.idx_to_classification_tags[tag_id]
        tag = Tag(self.classification_tags.get(tag_name), round(float(score), 4))
        tags = TagCollection([tag])
        return Annotation(ann.img_size, img_tags=tags)


def main():
    single_image_applier = ResnetSingleImageApplier()
    default_inference_mode_config = InfModeFullImage.make_default_config(model_result_suffix='_resnet')
    dataset_applier = BatchInferenceApplier(single_image_inference=single_image_applier,
                                            default_inference_mode_config=default_inference_mode_config,
                                            config_validator=JsonConfigValidator())
    dataset_applier.run_inference()


if __name__ == '__main__':
    if os.getenv('DEBUG_LOG_TO_FILE', None):
        sly_logger.add_default_logging_into_file(logger, TaskPaths.DEBUG_DIR)
    sly.main_wrapper('RESNET_INFERENCE', main)
