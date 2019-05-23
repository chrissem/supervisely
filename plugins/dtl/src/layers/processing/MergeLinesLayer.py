# coding: utf-8

import math
from copy import copy, deepcopy

import numpy as np

from legacy_supervisely_lib.figure.figure_line import FigureLine
from Layer import Layer


START1_START2 = 0
START1_END2 = 1
END1_START2 = 2
END1_END2 = 3


def points_distance(point_a, point_b):
    return math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)


def vector_dot(vec_a, vec_b):
    return vec_a[0] * vec_b[0] + vec_a[1] * vec_b[1]


def vector_length(vec) -> float:
    return math.sqrt(vec[0]**2 + vec[1]**2)


def angle_between_vectors(vec_a, vec_b) -> float:
    d1 = vector_dot(vec_a, vec_b)
    d2 = vector_length(vec_a) * vector_length(vec_b)
    if d2 == 0:
        d2 = 0.0001
    angle = d1 / d2
    cangle = math.acos(min(max(angle, -1), 1))
    return math.degrees(cangle)


def line_to_vector(line):
    return [line[1][0] - line[0][0], line[1][1] - line[0][1]]


def angle_between_lines(line_a, line_b):
    return angle_between_vectors(line_to_vector(line_a), line_to_vector(line_b))


def get_all_distances(line_a, line_b):
    return [points_distance(a, b) for a, b in zip([line_a[0], line_a[0], line_a[-1], line_a[-1]],
                                                       [line_b[0], line_b[-1], line_b[0], line_b[-1]])]


def lines_nearest_connection(line_a, line_b):
    distances = get_all_distances(line_a, line_b)
    return distances.index(min(distances))


def angle_between_multilines(line_a, line_b, with_connection_type=False):
    connection_type = lines_nearest_connection(line_a, line_b)
    if connection_type == START1_START2:
        angle = angle_between_lines(line_a=[line_a[0], line_a[1]], line_b=[line_b[1], line_b[0]])
    elif connection_type == START1_END2:
        angle = angle_between_lines(line_a=[line_a[0], line_a[1]], line_b=[line_b[-2], line_b[-1]])
    elif connection_type == END1_START2:
        angle = angle_between_lines(line_a=[line_a[-2], line_a[-1]], line_b=[line_b[0], line_b[1]])
    elif connection_type == END1_END2:
        angle = angle_between_lines(line_a=[line_a[-1], line_a[-2]], line_b=[line_b[-2], line_b[-1]])
    if with_connection_type:
        return angle, connection_type
    else:
        return angle


def check_collinear_inner_rule(line_a, line_b):
    connection_type = lines_nearest_connection(line_a, line_b)
    if connection_type == START1_START2:
        return points_distance(line_a[1], line_a[0]) < points_distance(line_a[1], line_b[0]) and \
               points_distance(line_a[1], line_b[0]) > points_distance(line_a[0], line_b[0])

    elif connection_type == START1_END2:
        return points_distance(line_a[1], line_a[0]) < points_distance(line_a[1], line_b[-1]) and \
               points_distance(line_a[1], line_b[-1]) > points_distance(line_a[0], line_b[-1])

    elif connection_type == END1_START2:
        return points_distance(line_a[-2], line_a[-1]) < points_distance(line_a[-2], line_b[0]) and \
               points_distance(line_a[-2], line_b[0]) > points_distance(line_a[-1], line_b[0])

    elif connection_type == END1_END2:
        return points_distance(line_a[-2], line_a[-1]) < points_distance(line_a[-2], line_b[-1]) and \
               points_distance(line_a[-2], line_b[-1]) > points_distance(line_a[-1], line_b[-1])


def get_ortho_distance(line_a, point_b):
    a = points_distance(line_a[0], line_a[1])
    b = points_distance(line_a[0], point_b)
    c = points_distance(point_b, line_a[1])
    p = (a + b + c) / 2
    if a == 0:
        a = 0.001
    h = 2 * math.sqrt(abs(p * (p-a) * (p-b) * (p-c))) / a
    return h


def get_lines_ortho_distance(line_a, line_b):
    connection_type = lines_nearest_connection(line_a, line_b)
    if connection_type == START1_START2:
        return get_ortho_distance([line_a[1], line_a[0]], line_b[0])
    elif connection_type == START1_END2:
        return get_ortho_distance([line_a[1], line_a[0]], line_b[-1])
    elif connection_type == END1_START2:
        return get_ortho_distance([line_a[-2], line_a[-1]], line_b[0])
    elif connection_type == END1_END2:
        return get_ortho_distance([line_a[-2], line_a[-1]], line_b[-1])


def merge_two_lines(line_a, line_b):
    connection_type = lines_nearest_connection(line_a, line_b)
    if connection_type == START1_START2:
        return line_a[::-1] + line_b
    if connection_type == START1_END2:
        return line_b + line_a
    if connection_type == END1_START2:
        return line_a + line_b
    if connection_type == END1_END2:
        return line_a+ line_b[::-1]


def merge_lines_at_one_line_if_possible(a, b, maximal_angle, max_ends_distance, max_ortho_distance_coefficient):
    """
    Args:
        a: first line
        b: second line
        maximal_angle: maximal possible angle between 2 lines
        max_ends_distance: maximal possible distance between nearest ends of lines
        max_ortho_distance_coefficient: maximal possible orthogonal distance calculated as:
            ortho_distance < max_ends_distance * max_ortho_distance_coefficient and
            ortho_distance < nearest_ends_distance * max_ortho_distance_coefficient
            (recommended value: 0.3-0.7)
    Returns:
        one merged line or none
    """
    minimal_distance = min(get_all_distances(a, b))
    if minimal_distance <= max_ends_distance:
        if check_collinear_inner_rule(a, b):
            ortho_dist = get_lines_ortho_distance(a, b)
            if ortho_dist < minimal_distance * max_ortho_distance_coefficient:
                angle = angle_between_multilines(a, b)
                if angle <= maximal_angle:
                    return merge_two_lines(a, b)
    return None


def merge_lines_collinear(lines_list, maximal_angle, max_ends_distance, max_ortho_distance_coefficient):
    toMerge = copy(lines_list)
    resLines = []
    curInd = 0
    nextInd = 1

    while len(toMerge) > 0:
        curLine = toMerge[curInd]

        if len(toMerge) == 1:
            resLines.append(curLine)
            break

        nextLine = toMerge[nextInd]

        lineToAdd = merge_lines_at_one_line_if_possible(curLine,
                                                        nextLine,
                                                        maximal_angle,
                                                        max_ends_distance,
                                                        max_ortho_distance_coefficient)

        if lineToAdd is not None:
            del toMerge[curInd]
            del toMerge[nextInd - 1]

            toMerge.insert(0, lineToAdd)
            nextInd = 1
            curInd = 0
            continue
        else:
            nextInd = nextInd + 1
            if nextInd == len(toMerge):
                del toMerge[curInd]
                resLines.append(curLine)
                nextInd = 1

    return resLines


def merge_objects_lines(class_title, image_size_wh, figures, maximal_angle, max_ends_distance,
                        max_ortho_distance_coefficient):

    simple_lines = []

    for line in figures:
        packed = line.pack()
        points = packed['points']['exterior']
        simple_lines.append(points)

    result_lines = merge_lines_collinear(simple_lines, maximal_angle, max_ends_distance, max_ortho_distance_coefficient)

    results = []
    for line in result_lines:
        figure = FigureLine.from_np_points(class_title=class_title,
                                  image_size_wh=image_size_wh,
                                  exterior=line)
        results.extend(figure)

    return results


class MergeLinesLayer(Layer):

    action = 'merge_lines'

    layer_settings = {
        "required": ["settings"],
        "properties": {
            "settings": {
                "type": "object",
                "required": ["maximal_angle", "max_ends_distance", "max_ortho_distance_coefficient", "lines_class"],
                "properties": {
                    "maximal_angle": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 180
                    },
                    "max_ends_distance": {
                        "type": "string",
                        "pattern": "^[0-9]+(%)|(px)$"
                    },
                    "max_ortho_distance_coefficient": {
                        "type": "number",
                        "minimum": 0
                    },
                    "lines_class": {
                        "type": "string"
                    }
                }
            }
        }
    }

    def __init__(self, config):
        Layer.__init__(self, config)

    def process(self, data_el):
        img_desc, ann_orig = data_el
        ann = deepcopy(ann_orig)
        img_wh = ann_orig.image_size_wh
        image_mean_size = np.mean(img_wh)

        maximal_angle = self.settings['maximal_angle']
        max_ends_distance_str = self.settings['max_ends_distance']

        if 'px' in max_ends_distance_str:
            max_ends_distance = float(max_ends_distance_str[:-2])
        else:  # percent case
            max_ends_distance = float(max_ends_distance_str[:-1]) / 100.0 * image_mean_size

        max_ortho_distance_coefficient = self.settings['max_ortho_distance_coefficient']
        lines_class = self.settings.get('lines_class')

        all_figures = ann['objects']

        figures_result = []
        figures_for_processing = []

        for figure in all_figures:
            if not isinstance(figure, FigureLine):
                figures_result.append(figure)
            else:
                class_title = figure.class_title

                if lines_class == class_title:
                    figures_for_processing.append(figure)
                else:
                    figures_result.append(figure)


        figures_result.extend(merge_objects_lines(class_title=lines_class,
                                                  image_size_wh=img_wh,
                                                  figures=figures_for_processing,
                                                  maximal_angle=maximal_angle,
                                                  max_ends_distance=max_ends_distance,
                                                  max_ortho_distance_coefficient=max_ortho_distance_coefficient))

        ann['objects'] = figures_result
        yield img_desc, ann
