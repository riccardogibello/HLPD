import torch
import numpy as np
import networkx as nx


def create_hiera_distance_tensor(nodes, parent_child_map):
    n = len(nodes)
    nodes_with_root = ["root"]
    nodes_with_root.extend(nodes)
    tensor = np.zeros((n + 1, n + 1))
    non_root_labels = [item for items in parent_child_map.values() for item in items]
    root_children = []
    for key in parent_child_map.keys():
        if key not in non_root_labels:
            root_children.append(key)
    G = nx.DiGraph()
    for c in root_children:
        G.add_edge("root", c)
        G.add_edge(c, "root")
    for parent, children in parent_child_map.items():
        for child in children:
            G.add_edge(parent, child)
            G.add_edge(child, parent)
    shortest_paths = dict(nx.all_pairs_shortest_path_length(G))
    for i in range(n + 1):
        for j in range(i, n + 1):
            if i == j:
                tensor[i, j] = 1
            else:
                distance = shortest_paths.get(nodes_with_root[i], {}).get(nodes_with_root[j], float('inf'))
                if distance != float('inf'):
                    tensor[i, j] = distance + 1
                    tensor[j, i] = distance + 1
    no_root = torch.tensor(tensor[1:, 1:])
    return no_root


def create_relationship_matrix(labels, parent_child_relationship):
    matrix = [[0] * len(labels) for _ in range(len(labels))]
    label_index = {label: idx for idx, label in enumerate(labels)}
    non_root_labels = [item for items in parent_child_relationship.values() for item in items]
    root_children = []
    for key in parent_child_relationship.keys():
        if key not in non_root_labels:
            root_children.append(key)
    for i in range(len(root_children)):
        for j in range(len(root_children)):
            matrix[label_index[root_children[i]]][label_index[root_children[j]]] = 1
    for parent, children in parent_child_relationship.items():
        parent_idx = label_index[parent]
        for child in children:
            child_idx = label_index[child]
            matrix[parent_idx][child_idx] = 1
            matrix[child_idx][parent_idx] = 1
            siblings = parent_child_relationship.get(parent, [])
            for sibling in siblings:
                sibling_idx = label_index[sibling]
                matrix[child_idx][sibling_idx] = 1
                matrix[sibling_idx][child_idx] = 1
            queue = [child]
            while queue:
                current_label = queue.pop(0)
                for sibling in parent_child_relationship.get(current_label, []):
                    sibling_idx = label_index[sibling]
                    matrix[parent_idx][sibling_idx] = 1
                    matrix[sibling_idx][parent_idx] = 1  # Also mark inverse relationship
                    queue.append(sibling)
    for i in range(len(labels)):
        matrix[i][i] = 1
    return torch.tensor(matrix)


def only_ancestors_attention(labels, parent_child_relationship):
    matrix = [[0] * len(labels) for _ in range(len(labels))]
    label_index = {label: idx for idx, label in enumerate(labels)}
    for parent, children in parent_child_relationship.items():
        parent_idx = label_index[parent]
        for child in children:
            child_idx = label_index[child]
            matrix[child_idx][parent_idx] = 1
            queue = [child]
            while queue:
                current_label = queue.pop(0)
                for sibling in parent_child_relationship.get(current_label, []):
                    sibling_idx = label_index[sibling]
                    matrix[sibling_idx][parent_idx] = 1  # Also mark inverse relationship
                    queue.append(sibling)
    for i in range(len(labels)):
        matrix[i][i] = 1
    return torch.tensor(matrix)


def parent_attention(labels, parent_child_relationship):
    matrix = [[0] * len(labels) for _ in range(len(labels))]
    label_index = {label: idx for idx, label in enumerate(labels)}
    for parent, children in parent_child_relationship.items():
        parent_idx = label_index[parent]
        for child in children:
            child_idx = label_index[child]
            matrix[child_idx][parent_idx] = 1
    for i in range(len(labels)):
        matrix[i][i] = 1
    return torch.tensor(matrix)


def child_attention(labels, parent_child_relationship):
    matrix = [[0] * len(labels) for _ in range(len(labels))]
    label_index = {label: idx for idx, label in enumerate(labels)}
    for parent, children in parent_child_relationship.items():
        parent_idx = label_index[parent]
        for child in children:
            child_idx = label_index[child]
            matrix[parent_idx][child_idx] = 1
    for i in range(len(labels)):
        matrix[i][i] = 1
    return torch.tensor(matrix)


def parent_and_child_attention(labels, parent_child_relationship):
    matrix = [[0] * len(labels) for _ in range(len(labels))]
    label_index = {label: idx for idx, label in enumerate(labels)}
    for parent, children in parent_child_relationship.items():
        parent_idx = label_index[parent]
        for child in children:
            child_idx = label_index[child]
            matrix[parent_idx][child_idx] = 1
            matrix[child_idx][parent_idx] = 1
    for i in range(len(labels)):
        matrix[i][i] = 1
    return torch.tensor(matrix)


def parent_and_child_and_siblings_attention(labels, parent_child_relationship):
    matrix = [[0] * len(labels) for _ in range(len(labels))]
    label_index = {label: idx for idx, label in enumerate(labels)}
    non_root_labels = [item for items in parent_child_relationship.values() for item in items]
    root_children = []
    for key in parent_child_relationship.keys():
        if key not in non_root_labels:
            root_children.append(key)
    for i in range(len(root_children)):
        for j in range(len(root_children)):
            matrix[label_index[root_children[i]]][label_index[root_children[j]]] = 1
    for parent, children in parent_child_relationship.items():
        parent_idx = label_index[parent]
        for child in children:
            child_idx = label_index[child]
            matrix[parent_idx][child_idx] = 1
            matrix[child_idx][parent_idx] = 1
            siblings = parent_child_relationship.get(parent, [])
            for sibling in siblings:
                sibling_idx = label_index[sibling]
                matrix[child_idx][sibling_idx] = 1
                matrix[sibling_idx][child_idx] = 1
    for i in range(len(labels)):
        matrix[i][i] = 1
    return torch.tensor(matrix)


def siblings_and_descendants_attention(labels, parent_child_relationship):
    matrix = [[0] * len(labels) for _ in range(len(labels))]
    label_index = {label: idx for idx, label in enumerate(labels)}
    non_root_labels = [item for items in parent_child_relationship.values() for item in items]
    root_children = []
    for key in parent_child_relationship.keys():
        if key not in non_root_labels:
            root_children.append(key)
    for i in range(len(root_children)):
        for j in range(len(root_children)):
            matrix[label_index[root_children[i]]][label_index[root_children[j]]] = 1
    for parent, children in parent_child_relationship.items():
        parent_idx = label_index[parent]
        for child in children:
            child_idx = label_index[child]
            siblings = parent_child_relationship.get(parent, [])
            for sibling in siblings:
                sibling_idx = label_index[sibling]
                matrix[child_idx][sibling_idx] = 1
                matrix[sibling_idx][child_idx] = 1
            queue = [child]
            while queue:
                current_label = queue.pop(0)
                for sibling in parent_child_relationship.get(current_label, []):
                    sibling_idx = label_index[sibling]
                    matrix[parent_idx][sibling_idx] = 1
                    queue.append(sibling)
    for i in range(len(labels)):
        matrix[i][i] = 1
    return torch.tensor(matrix)


def siblings_attention(labels, parent_child_relationship):
    matrix = [[0] * len(labels) for _ in range(len(labels))]
    label_index = {label: idx for idx, label in enumerate(labels)}
    non_root_labels = [item for items in parent_child_relationship.values() for item in items]
    root_children = []
    for key in parent_child_relationship.keys():
        if key not in non_root_labels:
            root_children.append(key)
    for parent, children in parent_child_relationship.items():
        parent_idx = label_index[parent]
        for child in children:
            child_idx = label_index[child]
            siblings = parent_child_relationship.get(parent, [])
            for sibling in siblings:
                sibling_idx = label_index[sibling]
                matrix[child_idx][sibling_idx] = 1
                matrix[sibling_idx][child_idx] = 1
            queue = [child]
            while queue:
                current_label = queue.pop(0)
                for sibling in parent_child_relationship.get(current_label, []):
                    sibling_idx = label_index[sibling]
                    matrix[parent_idx][sibling_idx] = 1
                    queue.append(sibling)
    for i in range(len(labels)):
        matrix[i][i] = 1
    return torch.tensor(matrix)


def batchify_attention_mask(attention_mask, batch_size):
    batched_attention_mask = attention_mask.unsqueeze(0).expand(batch_size, -1, -1)
    return batched_attention_mask


if __name__ == '__main__':
    labels = ["A", "B", "C", "D", "A.1", "A.2", "B.1", "C.1", "D.1", "B.2", "C.2", "D.2", "A.1.1", "B.1.1", "C.1.1",
              "D.1.1", "A.1.1.1", "A.1.1.1.1"]
    parent_childs = {
        'A': ['A.1', 'A.2'],
        'B': ['B.1', 'B.2'],
        'C': ['C.1', 'C.2'],
        'D': ['D.1', 'D.2'],
        'A.1': ['A.1.1'],
        'B.1': ['B.1.1'],
        'C.1': ['C.1.1'],
        'D.1': ['D.1.1'],
        'A.1.1': ['A.1.1.1'],
        'A.1.1.1': ['A.1.1.1.1']
    }
    print(torch.nn.Softmax(dim=1)(1 - create_hiera_distance_tensor(labels, parent_childs)))
