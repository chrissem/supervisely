# coding: utf-8

from copy import deepcopy

from Layer import Layer

import supervisely_lib as sly


class SuperviselyLayer(Layer):

    action = 'supervisely'

    layer_settings = {
        "required": ["settings"],
        "properties": {
            "settings": {}
        }
    }

    def __init__(self, config, output_folder, net):
        Layer.__init__(self, config)
        self.output_folder = output_folder
        self.net = net
        self.out_project = sly.Project(directory=output_folder, mode=sly.OpenMode.CREATE)
        self.net_change_images = self.net.may_require_images()

    def is_archive(self):
        return False

    def validate_dest_connections(self):
        for dst in self.dsts:
            if len(dst) == 0:
                raise ValueError("Destination name in '{}' layer is empty!".format(self.action))

    def process(self, data_el):
        img_desc, ann = data_el

        ann_to_save = deepcopy(ann)
        ann_to_save.normalize_figures()
        packed_ann = ann_to_save.pack()

        dataset_name = img_desc.get_res_ds_name()
        if not self.out_project.datasets.has_key(dataset_name):
            self.out_project.create_dataset(dataset_name)
        out_dataset = self.out_project.datasets.get(dataset_name)

        out_item_name = self.net.get_free_name(img_desc) + img_desc.get_image_ext()
        if self.net_change_images:
            if img_desc.need_write():
                out_dataset.add_item_np(out_item_name, img_desc.image_data, ann=packed_ann)
            else:
                out_dataset.add_item_file(out_item_name, img_desc.get_img_path(), ann=packed_ann)

        yield ([img_desc, ann],)
