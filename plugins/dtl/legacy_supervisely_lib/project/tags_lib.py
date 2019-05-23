# coding: utf-8
import enum

from copy import deepcopy
from typing import Iterable


class TagMeta:
    @enum.unique
    class TagFormat(enum.Enum):
        STR = 1
        DICT = 2

    # These match the serialized JSON values.
    VALUE_TYPE_ANY_NUMBER = 'any_number'
    VALUE_TYPE_ANY_STRING = 'any_string'
    VALUE_TYPE_NONE = 'none'
    VALUE_TYPE_ONEOF_STRING = 'oneof_string'

    def __init__(self, json_data):
        self._json_data = deepcopy(json_data)
        self._validate()

    def _validate(self):
        _ = self.format
        _ = self.name
        if self.value_type == TagMeta.VALUE_TYPE_ONEOF_STRING:
            _ = self.values

    def _get_dict_field(self, field_name, default_value=None):
        if self.format != TagMeta.TagFormat.DICT:
            raise ValueError('Field {} is only available in tags stored as dicts.'.format(field_name))
        return self._json_data.get(field_name, default_value)

    @property
    def format(self):
        if isinstance(self._json_data, str):
            return TagMeta.TagFormat.STR
        elif isinstance(self._json_data, dict):
            return TagMeta.TagFormat.DICT
        else:
            raise ValueError('Unsupported tag JSON data type: {}'.format(type(self._json_data)))

    @property
    def _is_dict(self):
        return isinstance(self._json_data, dict)

    @property
    def name(self):
        name_getters = {
            TagMeta.TagFormat.STR: lambda x: x,
            TagMeta.TagFormat.DICT: lambda x: x['name'],
        }
        return name_getters[self.format](self._json_data)

    @property
    def color(self):
        return self._get_dict_field('color')

    @property
    def value_type(self):
        if self.format == TagMeta.TagFormat.STR:
            return TagMeta.VALUE_TYPE_NONE
        else:
            return self._get_dict_field('value_type', TagMeta.VALUE_TYPE_ANY_STRING)

    @property
    def values(self):
        if self.format == TagMeta.TagFormat.DICT and self.value_type == TagMeta.VALUE_TYPE_ONEOF_STRING:
            # If the value type is oneof, there must always be a values whitelist.
            # Copy to prevent modifications outside the class.
            return self._json_data['values'].copy()
        else:
            return None

    def add_value(self, value):
        if self.value_type != TagMeta.VALUE_TYPE_ONEOF_STRING:
            raise NotImplementedError('Adding a possible value to tag meta only supported for {} tags.'.format(TagMeta.VALUE_TYPE_ONEOF_STRING))
        if value in self.values:
            raise ValueError('Value option {} already exists for tag meta {}'.format(value, self.to_json()))
        self._json_data['values'].append(value)

    def is_valid_value(self, value):
        if self.value_type == TagMeta.VALUE_TYPE_NONE:
            return value is None
        elif self.value_type == TagMeta.VALUE_TYPE_ANY_STRING:
            return isinstance(value, str)
        elif self.value_type == TagMeta.VALUE_TYPE_ANY_NUMBER:
            return isinstance(value, int) or isinstance(value, float)
        elif self.value_type == TagMeta.VALUE_TYPE_ONEOF_STRING:
            return isinstance(value, str) and (value in self.values)
        else:
            raise ValueError('Unsupported tag value type: {}'.format(self.value_type))

    def to_json(self):
        return deepcopy(self._json_data)

    def __eq__(self, other):
        return (isinstance(other, TagMeta) and
                self.name == other.name and
                self.value_type == other.value_type and
                self.values == other.values)

    @staticmethod
    def from_tag_json(tag_json):
        if isinstance(tag_json, str):
            return TagMeta(tag_json)
        elif isinstance(tag_json, dict):
            # TODO factor out string constants.
            return TagMeta({'name': tag_json['name'], 'value_type': TagMeta.VALUE_TYPE_ANY_STRING })


class TagValue:
    def __init__(self, tag_meta: TagMeta, value=None):
        if not tag_meta.is_valid_value(value):
            raise ValueError('Value {} is not valid for given meta tag.'.format(value))
        self._tag_meta = tag_meta
        self._value = value

    @property
    def value(self):
        return self._value

    @property
    def meta(self):
        return self._tag_meta

    def to_json(self):
        if self._tag_meta.format == TagMeta.TagFormat.STR:
            return self._tag_meta.name
        else:
            result_dict = {'name': self._tag_meta.name}
            if self._value is not None:
                result_dict['value'] = self._value
            return result_dict


def tag_name_from_json(tag_json):
    if isinstance(tag_json, str):
        return tag_json
    elif isinstance(tag_json, dict):
        return tag_json['name']
    raise ValueError('Only string and dict tags are supported. Instead got {}: {}.'.format(type(tag_json), tag_json))


def tag_value_from_json(tag_json):
    if isinstance(tag_json, str):
        return None
    elif isinstance(tag_json, dict):
        return tag_json.get('value', None)
    raise ValueError('Only string and dict tags are supported. Instead got {}: {}.'.format(type(tag_json), tag_json))


class TagMetaCollection:
    def __init__(self, tag_metas: Iterable[TagMeta]):
        self._tags_meta = {}
        for tag_meta in tag_metas:
            if tag_meta.name in self._tags_meta:
                raise ValueError('Received multiple tag meta entries with the same name: {}'.format(tag_meta.name))
            self._tags_meta[tag_meta.name] = tag_meta

    @staticmethod
    def _check_old_new_tag_meta_matches(old_tag_meta: TagMeta, new_tag_meta: TagMeta):
        if old_tag_meta != new_tag_meta:
            raise ValueError('Old meta information does not match new meta for tag {}'.format(new_tag_meta.name))

    @property
    def names(self):
        return self._tags_meta.keys()

    def get_tag_meta_by_name(self, name):
        return self._tags_meta.get(name, None)

    def _check_intersecting_tag_values_match(self, other: Iterable[TagMeta]):
        for other_value in other:
            old_tag_meta = self._tags_meta.get(other_value.name, None)
            if old_tag_meta is not None:
                self._check_old_new_tag_meta_matches(old_tag_meta, other_value)

    def update(self, new_tags_meta: Iterable[TagMeta]):
        new_tags_meta = list(new_tags_meta)
        self._check_intersecting_tag_values_match(new_tags_meta)
        self._tags_meta.update({new_tag_meta.name: new_tag_meta for new_tag_meta in new_tags_meta if
                                new_tag_meta.name not in self._tags_meta})

    def intersection(self, new_tags_meta: Iterable[TagMeta]):
        new_tags_meta = list(new_tags_meta)
        self._check_intersecting_tag_values_match(new_tags_meta)
        common_tags = [self._tags_meta[tag_meta.name] for tag_meta in new_tags_meta if tag_meta.name in self._tags_meta]
        return TagMetaCollection(common_tags)

    def difference(self, other_tags_meta: Iterable[TagMeta]):
        other_tags_meta = list(other_tags_meta)
        self._check_intersecting_tag_values_match(other_tags_meta)
        other_names = set(tag_meta.name for tag_meta in other_tags_meta)
        return TagMetaCollection([tag_meta for tag_meta in self.to_list() if tag_meta.name not in other_names])

    def to_list(self):
        return self._tags_meta.values()

    def to_json(self):
        return [tag_meta.to_json() for tag_meta in self._tags_meta.values()]

    @staticmethod
    def from_json(json_data):
        return TagMetaCollection(tag_metas=[TagMeta(tag_meta_json) for tag_meta_json in json_data])


# TODO generic container unified with TagMetaCollection?
class TagValueCollection:
    def __init__(self, tag_values: Iterable[TagValue]):
        self._tag_values = {tag_value.meta.name: tag_value for tag_value in tag_values}

    @staticmethod
    def tag_value_dict_from_iterable(tag_values: Iterable[TagValue]):
        return {tag_value.meta.name: tag_value for tag_value in tag_values}

    @staticmethod
    def check_old_new_tag_value_matches(old_value: TagValue, new_value: TagValue):
        if old_value.value != new_value.value:
            raise ValueError('Old value does not match new value for tag {}'.format(old_value.meta.name))

    def _check_intersecting_tag_values_match(self, other: Iterable[TagValue]):
        for other_value in other:
            old_value = self._tag_values.get(other_value.meta.name, None)
            if old_value is not None:
                self.check_old_new_tag_value_matches(old_value, other_value)

    def has_value(self, name):
        return name in self._tag_values

    def get_value(self, name):
        return self._tag_values.get(name, None)

    def add(self, tag_value: TagValue):
        self.update([tag_value])

    def update(self, other: Iterable[TagValue]):
        other = list(other)
        self._check_intersecting_tag_values_match(other)
        self._tag_values.update(
            {tag_value.meta.name: tag_value for tag_value in other if tag_value.meta.name not in self._tag_values})

    def intersection(self, other: Iterable[TagValue]):
        other = list(other)
        self._check_intersecting_tag_values_match(other)
        return TagValueCollection(
            [self._tag_values[tag_value.meta.name] for tag_value in other if tag_value.meta.name in self._tag_values])

    def difference(self, other: Iterable[TagValue]):
        other = list(other)
        self._check_intersecting_tag_values_match(other)
        other_names = set(tag_value.meta.name for tag_value in other)
        return TagValueCollection(
            [tag_value for tag_value in self._tag_values.values() if tag_value.meta.name not in other_names])

    # TODO implement
    def remove_by_name(self, name):
        raise NotImplementedError()

    # TODO implement
    def remove_by_name_and_value(self, name, value):
        raise NotImplementedError()