import copy


def generate_template(parent_child_map, label_index_dict, label_descriptions):
    label_descriptions_hiera = copy.deepcopy(label_descriptions)
    child_parent = {v: k for k, values in parent_child_map.items() for v in values}
    for child, parent in child_parent.items():
        parent_desc = label_descriptions[label_index_dict[parent]]
        if child in parent_child_map:
            template = f"{label_descriptions[label_index_dict[child]]} is a sub-topic of {parent_desc}."
            label_descriptions_hiera[label_index_dict[child]] = template
        else:
            template = f"{label_descriptions[label_index_dict[child]]} is a sub-topic of {parent_desc}."
            label_descriptions_hiera[label_index_dict[child]] = template
        if parent not in child_parent.keys():
            template = parent_desc
            label_descriptions_hiera[label_index_dict[parent]] = template
    return label_descriptions_hiera


if __name__ == '__main__':
    # Example data
    parent_child_map = {
        "label1": ["child1"],
        "label2": ["child2", "child3"]
    }

    label_index_dict = {
        "label1": 0,
        "child1": 1,
        "child2": 2,
        "label2": 3,
        "child3": 4
    }

    label_descriptions = [
        "Label 1",
        "Child 1",
        "Child 2",
        "Label 2",
        "Child 3"]

    # Generate template
    hiera_desc = generate_template(parent_child_map, label_index_dict, label_descriptions)
    print(hiera_desc)
