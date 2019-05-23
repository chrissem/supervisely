# Usage:
# docker run --rm -ti \
#        -v /path-to/model:/sly_task_data/model
#        [model docker image name]
#        python -- /workdir/src/rest_inference.py

import os

from inference import ObjectDetectionSingleImageApplier

from supervisely_lib.nn.inference import rest_server
from supervisely_lib.nn.inference.rest_constants import REST_INFERENCE_PORT


if __name__ == '__main__':
    port = os.getenv(REST_INFERENCE_PORT, '')
    model = ObjectDetectionSingleImageApplier(task_model_config={})
    server = rest_server.RestInferenceServer(model=model, name=__name__, port=port)
    server.run()