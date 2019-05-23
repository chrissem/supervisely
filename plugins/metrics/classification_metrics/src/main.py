# coding: utf-8

import os

import supervisely_lib as sly
from supervisely_lib.metric.classification_metrics import ClassificationMetrics
from supervisely_lib.metric.common import check_tag_mapping, TAGS_MAPPING, CONFIDENCE_THRESHOLD
from supervisely_lib.io.json import load_json_file


def main():
    settings = load_json_file(sly.TaskPaths.SETTINGS_PATH)
    sly.logger.info('Input settings:', extra={'config': settings})

    metric = ClassificationMetrics(settings[TAGS_MAPPING], settings[CONFIDENCE_THRESHOLD])
    applier = sly.MetricProjectsApplier(metric, settings)
    check_tag_mapping(applier.project_gt, applier.project_pred, settings[TAGS_MAPPING])
    applier.run_evaluation()
    metric.log_total_metrics()


if __name__ == '__main__':
    if os.getenv('DEBUG_LOG_TO_FILE', None):
        sly.add_default_logging_into_file(sly.logger, sly.TaskPaths.DEBUG_DIR)
    sly.main_wrapper('METRIC_EVALUATION', main)
