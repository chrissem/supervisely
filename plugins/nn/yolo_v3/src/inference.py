# coding: utf-8

import os

from darknet_utils import load_net

import supervisely_lib as sly
from supervisely_lib.nn.hosted.inference_single_image import SingleImageInferenceBase, GPU_DEVICE
from supervisely_lib.nn.hosted.inference_batch import BatchInferenceApplier
from supervisely_lib.nn.hosted.inference_modes import InfModeFullImage, CONFIDENCE
import common
from yolo_config_utils import read_config, replace_config_section_values, write_config, MODEL_CFG, NET_SECTION


class YOLOSingleImageApplier(SingleImageInferenceBase):

    def __init__(self, task_model_config=None):
        sly.logger.info('YOLOv3 inference init started.')
        super().__init__(task_model_config)
        self.confidence_thresh = self._config['min_confidence_threshold']
        sly.logger.info('YOLOv3 inference init done.')

    @staticmethod
    def get_default_config():
        return {
            'min_confidence_threshold': 0.5,
            GPU_DEVICE: 0,
            'confidence_tag_name': CONFIDENCE
        }

    @property
    def train_classes_key(self):
        return common.train_classes_key()

    @property
    def class_title_to_idx_key(self):
        return common.class_to_idx_config_key()

    def _model_out_tags(self):
        return sly.TagMetaCollection(items=[self.confidence_tag_meta])

    def _load_train_config(self):
        self.confidence_tag_meta = sly.TagMeta(self._config['confidence_tag_name'], sly.TagValueType.ANY_NUMBER)
        super()._load_train_config()

    def _validate_model_config(self, config):
        common.YoloJsonConfigValidator().validate_inference_cfg(config)

    def _construct_and_fill_model(self):
        super()._construct_and_fill_model()
        self.device_ids = sly.env.remap_gpu_devices([self._config[GPU_DEVICE]])

        yolo_config = read_config(os.path.join(sly.TaskPaths.MODEL_DIR, MODEL_CFG))
        [net_config] = [section for section in yolo_config if section.name == NET_SECTION]
        net_overrides = {
            'batch': 1,
            'subdivisions': 1
        }
        replace_config_section_values(net_config, net_overrides)
        effective_model_cfg_path = os.path.join('/tmp', MODEL_CFG)
        write_config(yolo_config, effective_model_cfg_path)
        self.model = load_net(effective_model_cfg_path.encode('utf-8'),
                              os.path.join(sly.TaskPaths.MODEL_DIR, 'model.weights').encode('utf-8'), 0)
        sly.logger.info('Weights are loaded.')

    def inference(self, img, ann):
        labels = common.infer_on_image(image=img,
                                       model=self.model,
                                       idx_to_class=self.out_class_mapping,
                                       confidence_thresh=self.confidence_thresh,
                                       confidence_tag_meta=self.confidence_tag_meta,
                                       num_classes=len(self.class_title_to_idx))
        return sly.Annotation(ann.img_size, labels=labels)


def main():
    single_image_applier = YOLOSingleImageApplier()
    default_inference_mode_config = InfModeFullImage.make_default_config(model_result_suffix='_yolo')
    dataset_applier = BatchInferenceApplier(single_image_inference=single_image_applier,
                                            default_inference_mode_config=default_inference_mode_config,
                                            config_validator=common.YoloJsonConfigValidator())
    dataset_applier.run_inference()


if __name__ == '__main__':
    if os.getenv('DEBUG_LOG_TO_FILE', None):
        sly.add_default_logging_into_file(sly.logger, sly.TaskPaths.DEBUG_DIR)
    sly.main_wrapper('YOLO_V3_INFERENCE', main)
