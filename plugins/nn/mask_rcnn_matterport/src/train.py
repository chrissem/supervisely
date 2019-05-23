# coding: utf-8

import os
import os.path as osp

import custom_config
import model as modellib

from custom_dataset import MaskRCNNSuperviselyDataset

import supervisely_lib as sly
import supervisely_lib.nn.dataset
from supervisely_lib import logger
from supervisely_lib.nn.config import JsonConfigValidator
from supervisely_lib.nn.hosted.class_indexing import CONTINUE_TRAINING, TRANSFER_LEARNING
from supervisely_lib.nn.hosted.trainer import SuperviselyModelTrainer


class MaskRCNNTrainer(SuperviselyModelTrainer):

    @staticmethod
    def get_default_config():
        return {
            'dataset_tags': {
                'train': 'train',
                'val': 'val',
            },
            'batch_size': {
                'train': 1,
                'val': 1,
            },
            'input_size': {
                'min_dim': 256,
                'max_dim': 256,
                'width': 0,  # Not used. For compatibility.
                'height': 0  # Not used. For compatibility.
            },
            'special_classes': {
                'background': 'bg',
            },
            'epochs': 2,
            'lr': 0.001,
            'weights_init_type': TRANSFER_LEARNING,  # CONTINUE_TRAINING,
            'gpu_devices': [0],
            'train_layers': 'all',  # Options: 'all', '3+', '4+', '5+', 'heads'
        }

    def __init__(self):
        self.bkg_input_idx = 0
        super().__init__(default_config=MaskRCNNTrainer.get_default_config())
        logger.info('Model is ready to train.')

    @property
    def class_title_to_idx_key(self):
        return custom_config.class_to_idx_config_key()

    @property
    def train_classes_key(self):
        return custom_config.train_classes_key()

    def _validate_train_cfg(self, config):
        JsonConfigValidator().validate_train_cfg(config)

    def _determine_model_classes(self):
        self._determine_model_classes_segmentation(bkg_input_idx=self.bkg_input_idx)

    def _construct_and_fill_model(self):
        gpu_device_ids = sly.env.remap_gpu_devices(self.config['gpu_devices'])
        src_size = self.config['input_size']
        self.input_size_limits = (src_size['min_dim'], src_size['max_dim'])

        train_len = len(self.datasets['train'].samples)
        val_len = len(self.datasets['val'].samples)
        train_steps = train_len // self.config['batch_size']['train']
        val_steps = val_len // self.config['batch_size']['val']

        self.mask_rcnn_config = custom_config.make_config(
            n_classes=max(self.class_title_to_idx.values()) + 1,
            size=self.input_size_limits,
            train_steps=train_steps,
            val_steps=val_steps,
            gpu_count=len(gpu_device_ids),
            lr=self.config['lr'],
            batch_size=self.config['batch_size']['train'])

        self.model = modellib.MaskRCNN(mode="training",
                                       config=self.mask_rcnn_config,
                                       model_dir=sly.TaskPaths.RESULTS_DIR)

        weights_init_type = self.config['weights_init_type']
        model_weights_file = osp.join(sly.TaskPaths.MODEL_DIR, 'model_weights', 'model.h5')

        if weights_init_type == TRANSFER_LEARNING:
            self.model.load_weights(model_weights_file, by_name=True, exclude=["mrcnn_class_logits",
                                                                               "mrcnn_bbox_fc",
                                                                               "mrcnn_bbox",
                                                                               "mrcnn_mask"])
            logger.info('transfer_learning mode. Last layers were initialized randomly.')
        elif weights_init_type == CONTINUE_TRAINING:
            self.model.load_weights(model_weights_file, by_name=True)
            logger.info('continue_training mode. All layers have been initialized from previous snapshot.')

    def _construct_loss(self):
        # No external loss object, everything is encapsulated in the model training logic.
        pass

    def _construct_data_loaders(self):
        self.datasets = {}
        for the_name, the_tag in self.name_to_tag.items():
            samples_lst = self._deprecated_samples_by_tag[the_tag]
            supervisely_lib.nn.dataset.ensure_samples_nonempty(samples_lst, the_tag, self.project.meta)
            the_ds = MaskRCNNSuperviselyDataset(
                project_meta=self.project.meta,
                samples=samples_lst,
                class_mapping=self.class_title_to_idx
            )
            self.datasets[the_name] = the_ds
            logger.info('Prepared dataset.', extra={
                'dataset_purpose': the_name, 'dataset_tag': the_tag, 'sample_cnt': len(samples_lst)
            })

    def train(self):
        self.model.train(self.datasets['train'], self.datasets['val'],
                         learning_rate=self.mask_rcnn_config.LEARNING_RATE,
                         epochs=self.config['epochs'],
                         layers=self.config['train_layers'],
                         out_config=self.out_config,
                         sly_checkpoints_saver=self.checkpoints_saver)


def main():
    x = MaskRCNNTrainer()  # load model & prepare all
    x.train()


if __name__ == '__main__':
    if os.getenv('DEBUG_LOG_TO_FILE', None):
        sly.sly_logger.add_default_logging_into_file(logger, sly.TaskPaths.DEBUG_DIR)
    sly.main_wrapper('MASK_RCNN_MATTERPORT_TRAIN', main)
