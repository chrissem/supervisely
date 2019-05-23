# coding: utf-8

import requests
from collections import namedtuple
from copy import deepcopy

from supervisely_lib._utils import camel_to_snake


class ApiField:
    ID =                'id'
    NAME =              'name'
    DESCRIPTION =       'description'
    CREATED_AT =        'createdAt'
    UPDATED_AT =        'updatedAt'
    ROLE =              'role'
    TEAM_ID =           'teamId'
    WORKSPACE_ID =      'workspaceId'
    CONFIG =            'config'
    SIZE =              'size'
    PLUGIN_ID =         'pluginId'
    PLUGIN_VERSION =    'pluginVersion'
    HASH =              'hash'
    STATUS =            'status'
    ONLY_TRAIN =        'onlyTrain'
    USER_ID =           'userId'
    README =            'readme'
    PROJECT_ID =        'projectId'
    IMAGES_COUNT =      'imagesCount'
    DATASET_ID =        'datasetId'
    LINK =              'link'
    WIDTH =             'width'
    HEIGHT =            'height'
    WEIGHTS_LOCATION =  'weightsLocation'
    IMAGE_ID =          'imageId'
    IMAGE_NAME =        'imageName'
    ANNOTATION =        'annotation'
    TASK_ID =           'taskId'
    LABELS_COUNT =      'labelsCount'
    FILTER =            'filter'
    META =              'meta'
    SHARED_LINK =       'sharedLinkToken'
    EXPLORE_PATH =      'explorePath'
    MIME =              'mime'
    EXT =               'ext'
    TYPE =              'type'
    DEFAULT_VERSION =   'defaultVersion'
    DOCKER_IMAGE =      'dockerImage'
    CONFIGS =           'configs'
    VERSIONS =          'versions'
    VERSION =           'version'
    TOKEN =             'token'
    CAPABILITIES =      'capabilities'
    STARTED_AT =        'startedAt'
    FINISHED_AT =       'finishedAt'
    AGENT_ID =          'agentId'
    MODEL_ID =          'modelId'
    RESTART_POLICY =    'restartPolicy'
    SETTINGS =          'settings'
    SORT =              'sort'
    SORT_ORDER =        'sort_order'
    IMAGES =            'images'
    IMAGE_IDS =         'imageIds'
    ANNOTATIONS =       'annotations'


def _get_single_item(items):
    if len(items) == 0:
        return None
    if len(items) > 1:
        raise RuntimeError('There are several items with the same name {!r}')
    return items[0]


class _JsonConvertibleModule:
    def _convert_json_info(self, info: dict):
        raise NotImplementedError()


class ModuleApiBase(_JsonConvertibleModule):
    MAX_WAIT_ATTEMPTS = 999

    @staticmethod
    def info_sequence():
        raise NotImplementedError()

    @staticmethod
    def info_tuple_name():
        raise NotImplementedError()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        try:
            cls.InfoType = namedtuple(cls.info_tuple_name(), [camel_to_snake(name) for name in cls.info_sequence()])
        except NotImplementedError:
            pass

    def __init__(self, api):
        self._api = api

    def _add_sort_param(self, data):
        results = deepcopy(data)
        results[ApiField.SORT] = ApiField.ID
        results[ApiField.SORT_ORDER] = 'asc'  # @TODO: move to enum
        return results

    def get_list_all_pages(self, method, data, progress_cb=None):
        data = self._add_sort_param(data)
        first_response = self._api.post(method, data).json()
        total = first_response['total']
        per_page = first_response['perPage']
        pages_count = first_response['pagesCount']

        results = first_response['entities']
        if progress_cb is not None:
            progress_cb(len(results))
        if pages_count == 1 and len(first_response['entities']) == total:
            pass
        else:
            for page_idx in range(2, pages_count + 1):
                temp_resp = self._api.post(method, {**data, 'page': page_idx, 'per_page': per_page})
                temp_items = temp_resp.json()['entities']
                results.extend(temp_items)
                if progress_cb is not None:
                    progress_cb(len(temp_items))
            if len(results) != total:
                raise RuntimeError('Method {!r}: error during pagination, some items are missed'.format(method))

        return [self._convert_json_info(item) for item in results]

    @staticmethod
    def _get_info_by_name(get_info_by_filters_fn, name):
        filters = [{"field": ApiField.NAME, "operator": "=", "value": name}]
        return get_info_by_filters_fn(filters)

    def get_info_by_id(self, id):
        raise NotImplementedError()

    @staticmethod
    def _get_free_name(exist_check_fn, name):
        res_title = name
        suffix = 1
        while exist_check_fn(res_title):
            res_title = '{}_{:03d}'.format(name, suffix)
            suffix += 1
        return res_title

    def _convert_json_info(self, info: dict):
        if info is None:
            return None
        else:
            return self.InfoType(*[info[field_name] for field_name in self.info_sequence()])

    def _get_info_by_id(self, id, method):
        try:
            response = self._api.post(method, {ApiField.ID: id})
        except requests.exceptions.HTTPError as error:
            if error.response.status_code == 404:
                return None
            else:
                raise error
        return self._convert_json_info(response.json())


# Base class for entities that have a parent object in the system.
class ModuleApi(ModuleApiBase):
    def __init(self, api):
        self._api = api

    def get_info_by_name(self, parent_id, name):
        return self._get_info_by_name(
            get_info_by_filters_fn=lambda module_name: self._get_info_by_filters(parent_id, module_name), name=name)

    def _get_info_by_filters(self, parent_id, filters):
        items = self.get_list(parent_id, filters)
        return _get_single_item(items)

    def get_list(self, parent_id, filters=None):
        raise NotImplementedError()

    def exists(self, parent_id, name):
        return self.get_info_by_name(parent_id, name) is not None

    def get_free_name(self, parent_id, name):
        return self._get_free_name(exist_check_fn=lambda module_name: self.exists(parent_id, module_name), name=name)

    def _get_effective_new_name(self, parent_id, name, change_name_if_conflict=False):
        return self.get_free_name(parent_id, name) if change_name_if_conflict else name


# Base class for entities that do not have a parent object in the system.
class ModuleNoParent(ModuleApiBase):
    def get_info_by_name(self, name):
        return self._get_info_by_name(get_info_by_filters_fn=self._get_info_by_filters, name=name)

    def _get_info_by_filters(self, filters):
        items = self.get_list(filters)
        return _get_single_item(items)

    def get_list(self, filters=None):
        raise NotImplementedError()

    def exists(self, name):
        return self.get_info_by_name(name) is not None

    def get_free_name(self, name):
        return self._get_free_name(exist_check_fn=lambda module_name: self.exists(module_name), name=name)

    def _get_effective_new_name(self, name, change_name_if_conflict=False):
        return self.get_free_name(name) if change_name_if_conflict else name


class CloneableModuleApi(ModuleApi):
    def _clone_api_method_name(self):
        raise NotImplementedError()

    def _clone(self, clone_type: dict, dst_workspace_id: int, dst_name: str):
        response = self._api.post(self._clone_api_method_name(), {**clone_type,
                                                                  ApiField.WORKSPACE_ID: dst_workspace_id,
                                                                  ApiField.NAME: dst_name})
        return response.json()[ApiField.TASK_ID]

    def clone(self, id, dst_workspace_id, dst_name):
        return self._clone({ApiField.ID: id}, dst_workspace_id, dst_name)

    def clone_by_shared_link(self, shared_link, dst_workspace_id, dst_name):
        return self._clone({ApiField.SHARED_LINK: shared_link}, dst_workspace_id, dst_name)

    def clone_from_explore(self, explore_path, dst_workspace_id, dst_name):
        return self._clone({ApiField.EXPLORE_PATH: explore_path}, dst_workspace_id, dst_name)

    def get_or_clone_from_explore(self, explore_path, dst_workspace_id, dst_name):
        if not self.exists(dst_workspace_id, dst_name):
            task_id = self.clone_from_explore(explore_path, dst_workspace_id, dst_name)
            self._api.task.wait(task_id, self._api.task.Status.FINISHED)
        item = self.get_info_by_name(dst_workspace_id, dst_name)
        return item


class ModuleWithStatus:
    def get_status(self, id):
        raise NotImplementedError()

    def raise_for_status(self, status):
        raise NotImplementedError()


class UpdateableModule(_JsonConvertibleModule):
    def __init__(self, api):
        self._api = api

    def _get_update_method(self):
        raise NotImplementedError()

    def update(self, id, name=None, description=None):
        if name is None and description is None:
            raise ValueError("\'name\' or \'description\' or both have to be specified")

        body = {ApiField.ID: id}
        if name is not None:
            body[ApiField.NAME] = name
        if description is not None:
            body[ApiField.DESCRIPTION] = description

        response = self._api.post(self._get_update_method(), body)
        return self._convert_json_info(response.json())
