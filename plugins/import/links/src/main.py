# coding: utf-8

import os
import requests
import json
from slugify import slugify

import supervisely_lib as sly


headers = {}
server_address = None
task_context = None
append_to_existing_project = None


def get_task_context(task_id):
    url = os.path.join(server_address, 'public/api/v2/tasks.context')
    response = requests.post(url, data={'id': task_id}, headers=headers)
    return response.json()


def get_existing_id(api_method, name, additional_filters=None):
    url = os.path.join(server_address, api_method)
    filter = [{'field': 'title', 'operator': '=', 'value': name}]
    filter.extend(additional_filters or [])
    response = requests.post(url, json={'groupId': task_context['team']['id'], 'filter': filter}, headers=headers)

    if response.json()['total'] == 0:
        return None
    else:
        return response.json()['entities'][0]['id']


def get_free_name(api_method, name, additional_filters=None):
    free_name = name
    cnt_attempts = 0
    while True:
        existing_id = get_existing_id(api_method, free_name, additional_filters)
        if existing_id is None:
            break
        cnt_attempts += 1
        free_name = "{}_{:03d}".format(name, cnt_attempts)
    return free_name


def create_project_api(project_name):
    if append_to_existing_project is True:
        project_id = get_existing_id('public/api/v2/projects.list', project_name)
        if project_id is None:
            raise RuntimeError("Project {!r} not found".format(project_name))

    else:
        free_project_name = get_free_name('public/api/v2/projects.list', project_name)
        response = requests.post(os.path.join(server_address, 'public/api/v2/projects.add'),
                                 data={'workspaceId': task_context['workspace']['id'], 'title': free_project_name},
                                 headers=headers)
        project_id = response.json()['id']
    return project_id


def create_dataset_api(project_id, dataset_name):
    project_id_filter = [{'field': 'projectId', 'operator': '=', 'value': project_id}]
    free_project_name = get_free_name('public/api/v2/datasets.list', dataset_name, project_id_filter)
    response = requests.post(os.path.join(server_address, 'public/api/v2/datasets.add'),
                             data={'projectId': project_id, 'title': free_project_name},
                             headers=headers)
    dataset_id = response.json()['id']
    return dataset_id


def add_image_to_dataset(dataset_id, img_title, link):
    dataset_id_filter = [{'field': 'datasetId', 'operator': '=', 'value': dataset_id}]
    free_image_name = get_free_name('public/api/v2/images.list', img_title, dataset_id_filter)

    response = requests.post(os.path.join(server_address, 'public/api/v2/images.remote.upsert'),
                             data={"link": link},
                             headers=headers)
    internal_link_id = response.json()['id']
    response = requests.post(os.path.join(server_address, 'public/api/v2/images.add'),
                             data={'datasetId': dataset_id,
                                   'title': free_image_name,
                                   'link': link},
                             headers=headers)
    image_id = response.json()['id']
    return image_id


def process_dataset_links(project_id, file_path):
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    dataset_id = create_dataset_api(project_id, dataset_name)
    with open(file_path) as fp:
        lines = fp.readlines()
        progress = sly.Progress('Import dataset: {}'.format(dataset_name), len(lines))
        for line in lines:
            url = line.strip()
            if url:
                try:
                    image_split_name = os.path.splitext(os.path.basename(url))
                    image_name = image_split_name[0]
                    image_name = slugify(image_name)
                    if len(image_split_name) == 2:
                        image_ext = image_split_name[1]
                        image_name = image_name + image_ext
                    add_image_to_dataset(dataset_id, image_name, url)
                except Exception as e:
                    exc_str = str(e)
                    sly.logger.warn('Input link skipped due to error: {}'.format(exc_str), exc_info=True, extra={
                        'exc_str': exc_str,
                        'file_path': file_path,
                        'link': line,
                    })
            progress.iter_done_report()


def main():
    global server_address, task_context, append_to_existing_project

    with open('/sly_task_data/task_settings.json') as json_file:
        task_settings = json.load(json_file)

    server_address = task_settings['server_address']
    headers['x-api-key'] = task_settings['api_token']
    task_context = get_task_context(task_settings['task_id'])
    append_to_existing_project = task_settings['append_to_existing_project']

    project_id = create_project_api(task_settings['res_names']['project'])

    data_dir = "/sly_task_data/data"
    files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    for file in files:
        file_path = os.path.join(data_dir, file)
        process_dataset_links(project_id, file_path)

    with open('/sly_task_data/results/project_info.json', 'w') as outfile:
        json.dump({'project_id': project_id}, outfile)


if __name__ == '__main__':
    sly.main_wrapper('IMAGES_ONLY_IMPORT', main)
