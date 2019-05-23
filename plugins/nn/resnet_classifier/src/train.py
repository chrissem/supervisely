# coding: utf-8

from collections import defaultdict

import os
import cv2
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

import supervisely_lib as sly
from supervisely_lib import logger
from supervisely_lib.nn.config import JsonConfigValidator
from supervisely_lib.nn.hosted.class_indexing import CONTINUE_TRAINING, TRANSFER_LEARNING
from supervisely_lib.nn.training.eval_planner import EvalPlanner
from supervisely_lib.nn.pytorch.weights import WeightsRW
from supervisely_lib.nn.hosted.trainer import SuperviselyModelTrainer
from supervisely_lib.nn.hosted.class_indexing import infer_training_class_to_idx_map
from supervisely_lib.task.progress import epoch_float

import config as config_lib
from common import create_classes, determine_resnet_model_configuration
from model_utils import create_model
from dataset import ResnetDataset
from metrics import Accuracy


# decrease lr after 'patience' calls w/out loss improvement
class LRPolicyWithPatience:
    def __init__(self, optim_cls, init_lr, patience, lr_divisor, model):
        self.optimizer = None
        self.optim_cls = optim_cls
        self.lr = init_lr
        self.patience = patience
        self.lr_divisor = lr_divisor
        self.losses = []
        self.last_reset_idx = 0

        logger.info('Selected optimizer.', extra={'optim_class': self.optim_cls.__name__})
        self._reset(model)

    def _reset(self, model):
        self.optimizer = self.optim_cls(model.parameters(), lr=self.lr)
        logger.info('Learning Rate has been updated.', extra={'lr': self.lr})

    def reset_if_needed(self, new_loss, model):
        self.losses.append(new_loss)
        no_recent_update = (len(self.losses) - self.last_reset_idx) > self.patience
        no_loss_improvement = min(self.losses[-self.patience:]) > min(self.losses)
        if no_recent_update and no_loss_improvement:
            self.lr /= float(self.lr_divisor)
            self._reset(model)
            self.last_reset_idx = len(self.losses)


class ResnetTrainer(SuperviselyModelTrainer):
    @staticmethod
    def get_default_config():
        return {
            'num_layers': 18,
            'dataset_tags': {
                'train': 'train',
                'val': 'val',
            },
            'batch_size': {
                'train': 2,
                'val': 2,
            },
            'data_workers': {
                'train': 0,
                'val': 0,
            },
            'allow_corrupted_samples': {
                'train': 0,
                'val': 0,
            },
            'input_size': {
                'width': 224,
                'height': 224,
            },
            'epochs': 2,
            'val_every': 0.5,
            'lr': 0.1,
            'lr_decreasing': {
                'patience': 1000,
                'lr_divisor': 5,
            },
            'weights_init_type': TRANSFER_LEARNING,  # CONTINUE_TRAINING,
            'gpu_devices': [0]
        }

    @property
    def classification_tags_key(self):
        return config_lib.classification_tags_key()

    @property
    def classification_tags_to_idx_key(self):
        return config_lib.classification_tags_to_idx_key()

    @property
    def class_title_to_idx_key(self):
        return config_lib.class_to_idx_config_key()

    @property
    def train_classes_key(self):
        return config_lib.train_classes_key()

    def _validate_train_cfg(self, config):
        JsonConfigValidator().validate_train_cfg(config)

    def _determine_model_classes(self):
        if 'classes' not in self.config:
            # Key-value tags are ignored as a source of class labels.
            img_tags = set(tag_meta.name for tag_meta in self.project.meta.tag_metas if
                           tag_meta.value_type == sly.TagValueType.NONE)
            img_tags -= set(self.config['dataset_tags'].values())
            train_classes = sorted(img_tags)
        else:
            train_classes = self.config['classes']

        if 'ignore_tags' in self.config:
            for tag in self.config['ignore_tags']:
                if tag in train_classes:
                    train_classes.remove(tag)

        if len(train_classes) < 2:
            raise RuntimeError('Training requires at least two input classes.')

        in_classification_tags_to_idx, self.classification_tags_sorted = create_classes(train_classes)
        self.classification_tags_to_idx = infer_training_class_to_idx_map(self.config['weights_init_type'],
                                                                          in_classification_tags_to_idx,
                                                                          sly.TaskPaths.MODEL_CONFIG_PATH,
                                                                          class_to_idx_config_key=self.classification_tags_to_idx_key)

        self.class_title_to_idx = {}
        self.out_classes = sly.ObjClassCollection()
        logger.info('Determined model internal class mapping', extra={'class_mapping': self.class_title_to_idx})
        logger.info('Determined model out classes', extra={'classes': self.classification_tags_sorted})

    def _determine_model_configuration(self):
        self.num_layers = determine_resnet_model_configuration(sly.TaskPaths.MODEL_CONFIG_PATH)

        # Check for possible num_layers field in old-style config. If exists, make sure it is consistent with
        # the actual model config and clear num_layers from training config before writing new-style model
        # config.
        training_model_config = self.config.get('num_layers', None)
        if training_model_config is not None:
            if training_model_config != self.num_layers:
                error_msg = (
                        'Unable to start training. num_layers in the training config is not consistent with ' +
                        'selected model architecture. Make sure you have selected the right model plugin and remove ' +
                        'num_layers from the training config as it is not required anymore.')
                logger.critical(error_msg,
                                extra={'training_model_config': self.config['num_layers'],
                                       'num_layers': self.num_layers})
                raise RuntimeError(error_msg)
            del self.config['num_layers']

    def _determine_out_config(self):
        super()._determine_out_config()
        self._determine_model_configuration()
        self.out_config[self.classification_tags_key] = [self.project.meta.tag_metas.get(cls).to_json()
                                                         for cls in self.classification_tags_sorted]
        self.out_config[self.classification_tags_to_idx_key] = self.classification_tags_to_idx
        self.out_config['num_layers'] = self.num_layers

    def _construct_and_fill_model(self):
        progress_dummy = sly.Progress('Building model:', 1)
        progress_dummy.iter_done_report()

        self.model = create_model(self.num_layers,
                                  n_cls=len(self.classification_tags_sorted), device_ids=self.device_ids)

        if sly.fs.dir_empty(sly.TaskPaths.MODEL_DIR):
            logger.info('Weights will not be inited.')
            # @TODO: add random init (m.weight.data.normal_(0, math.sqrt(2. / n))
        else:
            wi_type = self.config['weights_init_type']
            ewit = {'weights_init_type': wi_type}
            logger.info('Weights will be inited from given model.', extra=ewit)

            weights_rw = WeightsRW(sly.TaskPaths.MODEL_DIR)
            if wi_type == TRANSFER_LEARNING:
                self.model = weights_rw.load_for_transfer_learning(self.model, ignore_matching_layers=['fc'],
                                                                   logger=logger)
            elif wi_type == CONTINUE_TRAINING:
                self.model = weights_rw.load_strictly(self.model)

            logger.info('Weights are loaded.', extra=ewit)

    def _construct_loss(self):
        # neutral = self.neutral_input_idx
        self.metrics = {
            'accuracy': Accuracy(ignore_index=None)
        }
        self.criterion = CrossEntropyLoss()

        self.val_metrics = {
            'loss': self.criterion,
            **self.metrics
        }
        logger.info('Selected metrics.', extra={'metrics': list(self.metrics.keys())})

    def _construct_data_loaders(self):
        self.device_ids = sly.env.remap_gpu_devices(self.config['gpu_devices'])

        src_size = self.config['input_size']
        input_size = (src_size['height'], src_size['width'])

        self.pytorch_datasets = {}
        self.data_loaders = {}

        shuffle_drop_last = {
            'train': True,
            'val': False
        }

        for the_name, the_tag in self.name_to_tag.items():
            samples_lst = self._deprecated_samples_by_tag[the_tag]
            samples_count = len(samples_lst)
            # note that now batch_size from config determines batch for single device
            batch_sz = self.config['batch_size'][the_name]
            batch_sz_full = batch_sz * len(self.device_ids)

            if samples_count < batch_sz_full:
                raise RuntimeError('Project should contain at least '
                                   '{}(batch size) * {}(gpu devices) = {} samples tagged by "{}", '
                                   'but found {} samples.'
                                   .format(batch_sz, len(self.device_ids), batch_sz_full, the_name, samples_count))

            the_ds = ResnetDataset(
                project_meta=self.project.meta,
                samples=samples_lst,
                out_size=input_size,
                class_mapping=self.classification_tags_to_idx,
                out_classes=self.classification_tags_sorted,
                allow_corrupted_cnt=self.config['allow_corrupted_samples'][the_name],
                spec_tags=list(self.config['dataset_tags'].values())
            )
            self.pytorch_datasets[the_name] = the_ds
            logger.info('Prepared dataset.', extra={
                'dataset_purpose': the_name, 'dataset_tag': the_tag, 'sample_cnt': len(samples_lst)
            })

            n_workers = self.config['data_workers'][the_name]
            self.data_loaders[the_name] = DataLoader(
                dataset=self.pytorch_datasets[the_name],
                batch_size=batch_sz_full,  # it looks like multi-gpu validation works
                num_workers=n_workers,
                shuffle=shuffle_drop_last,
                drop_last=shuffle_drop_last[the_name]
            )
        logger.info('DataLoaders are constructed.')

        self.train_iters = len(self.data_loaders['train'])
        self.val_iters = len(self.data_loaders['val'])
        self.epochs = self.config['epochs']
        self.eval_planner = EvalPlanner(epochs=self.epochs, val_every=self.config['val_every'])

    def __init__(self):
        super().__init__(default_config=ResnetTrainer.get_default_config())

    def _dump_model_weights(self, out_dir):
        WeightsRW(out_dir).save(self.model)

    def _validation(self):
        logger.info("Before validation", extra={'epoch': self.epoch_flt})
        self.model.eval()
        metrics_values = defaultdict(int)
        samples_cnt = 0

        for val_it, (inputs, targets) in enumerate(self.data_loaders['val']):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = self.model(inputs)
            full_batch_size = inputs.size(0)
            for name, metric in self.val_metrics.items():
                metric_value = metric(outputs, targets)
                metric_value = metric_value.item()
                metrics_values[name] += metric_value * full_batch_size
            samples_cnt += full_batch_size

            logger.info("Validation in progress", extra={'epoch': self.epoch_flt,
                                                         'val_iter': val_it, 'val_iters': self.val_iters})

        for name in metrics_values:
            metrics_values[name] /= float(samples_cnt)

        sly.report_metrics_validation(self.epoch_flt, metrics_values)

        self.model.train()
        logger.info("Validation has been finished", extra={'epoch': self.epoch_flt})
        return metrics_values

    def train(self):
        progress = sly.Progress('Model training: ', self.epochs * self.train_iters)
        self.model.train()

        lr_decr = self.config['lr_decreasing']
        policy = LRPolicyWithPatience(
            optim_cls=Adam,
            init_lr=self.config['lr'],
            patience=lr_decr['patience'],
            lr_divisor=lr_decr['lr_divisor'],
            model=self.model
        )
        best_val_loss = float('inf')

        for epoch in range(self.epochs):
            logger.info("Before new epoch", extra={'epoch': self.epoch_flt})

            for train_it, (inputs_cpu, targets_cpu) in enumerate(self.data_loaders['train']):
                inputs, targets = inputs_cpu.requires_grad_().cuda(), targets_cpu.requires_grad_().cuda()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                policy.optimizer.zero_grad()
                loss.backward()
                policy.optimizer.step()

                metric_values_train = {'loss': loss.item()}
                for name, metric in self.metrics.items():
                    metric_values_train[name] = metric(outputs, targets)

                progress.iter_done_report()

                self.epoch_flt = epoch_float(epoch, train_it + 1, self.train_iters)
                sly.report_metrics_training(self.epoch_flt, metric_values_train)

                if self.eval_planner.need_validation(self.epoch_flt):
                    metrics_values_val = self._validation()
                    self.eval_planner.validation_performed()

                    val_loss = metrics_values_val['loss']
                    model_is_best = val_loss < best_val_loss
                    if model_is_best:
                        best_val_loss = val_loss
                        logger.info('It\'s been determined that current model is the best one for a while.')

                    self._save_model_snapshot(model_is_best, opt_data={
                        'epoch': self.epoch_flt,
                        'val_metrics': metrics_values_val,
                    })

                    policy.reset_if_needed(val_loss, self.model)

            logger.info("Epoch was finished", extra={'epoch': self.epoch_flt})


def main():
    cv2.setNumThreads(0)  # important for pytorch dataloaders
    x = ResnetTrainer()  # load model & prepare all
    x.train()


if __name__ == '__main__':
    if os.getenv('DEBUG_LOG_TO_FILE', None):
        sly.add_default_logging_into_file(sly.logger, sly.TaskPaths.DEBUG_DIR)
    sly.main_wrapper('RESNET_TRAIN', main)
