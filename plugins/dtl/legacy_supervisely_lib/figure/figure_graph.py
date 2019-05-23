# coding: utf-8

import json

from .abstract_figure import AbstractFigure

class FigureGraph(AbstractFigure):
    """Dummy class to support downloading graph objects in JSON format."""

    def __str__(self):
        return json.dumps(self.data, sort_keys=True, indent=4)

    # returns figure (w/out validation) or None; owns packed_obj
    @classmethod
    def from_packed(cls, packed_obj):
        return FigureGraph(packed_obj)

    # returns packed obj
    def pack(self):
        return self.data

    @property
    def tags(self):
        return self.data['tags']

    def normalize(self, img_size_wh):
        return [self]
