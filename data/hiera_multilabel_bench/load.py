import os
import shutil
from typing import Any
from datasets import load_dataset
import pandas as pd

from data.hiera_multilabel_bench.hiera_multilabel_bench import (
    AAPD_CONCEPTS,
    BGC_CONCEPTS,
    RCV_CONCEPTS,
    WOS_CONCEPTS,
)

from .hiera_label_descriptors import (
    label2desc_reduced_aapd,
    label2desc_reduced_bgc,
    label2desc_reduced_rcv,
)


def extract_tar_dataset(
    data_files_folder_path: str,
    dataset_name: str,
) -> str:
    folder_name = dataset_name.split("-")[0]
    new_folder_name = "_" + folder_name
    folder_path = os.path.join(
        data_files_folder_path,
        new_folder_name,
    )
    tar_file_path = os.path.join(
        data_files_folder_path,
        f"{folder_name}.tar.gz",
    )
    if not os.path.exists(tar_file_path):
        raise FileNotFoundError(
            f"Tar file {tar_file_path} not found. "
            "Please ensure the dataset is correctly placed."
        )

    if not os.path.exists(folder_path):
        print(f"Extracting {tar_file_path} to {folder_path}")
        import tarfile

        with tarfile.open(tar_file_path, "r:gz") as tar:
            tar.extractall(path=folder_path)

    # Remove from inside the extracted folder any file or folder that does not end with .jsonl
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if not item.endswith(".jsonl") and os.path.isfile(item_path):
            os.remove(item_path)
        elif not item.endswith(".jsonl") and os.path.isdir(item_path):
            shutil.rmtree(item_path)

    return folder_path


def custom_load_dataset(
    data_files_folder_path: str,
    dataset_name: str,
    split: str = "train",
) -> tuple[Any]:
    folder_name = dataset_name.split("-")[0]
    folder_path = os.path.join(
        data_files_folder_path,
        "_" + folder_name,
    )
    dataset_file_path = os.path.join(
        folder_path,
        f"{folder_name}.jsonl",
    )
    existing_dataset = os.path.exists(folder_path) and os.path.exists(dataset_file_path)
    if not existing_dataset:
        raise FileNotFoundError(
            f"Dataset file {dataset_file_path} not found. "
            "Please ensure the dataset is correctly extracted."
        )

    # Load the full dataset and filter by data_type
    ds = load_dataset(
        "json",
        data_files=dataset_file_path,
        split=None,
        cache_dir=folder_path,
    )
    # ds is a DatasetDict with a 'train' key by default when split=None
    if isinstance(ds, dict) and "train" in ds:
        ds = ds["train"]
    filtered_ds = ds.filter(lambda x: x.get("data_type", "train") == split)
    # Rename any columns containing "_concepts" substring to "concepts"
    filtered_ds = filtered_ds.rename_columns(
        {col: "concepts" for col in filtered_ds.column_names if "_concepts" in col}
    )
    return filtered_ds


def translate_into_structured_file(
    data_files_folder_path: str,
    dataset_name: str,
) -> pd.DataFrame:
    dataset_name = dataset_name.split("-")[0]
    name_hierarchy_map = {
        "wos": WOS_CONCEPTS,
        "bgc": BGC_CONCEPTS,
        "aapd": AAPD_CONCEPTS,
        # "rcv": RCV_CONCEPTS,
    }
    name_descriptions_map = {
        "wos": None,
        "bgc": label2desc_reduced_bgc,
        "aapd": label2desc_reduced_aapd,
        # "rcv": label2desc_reduced_rcv,
    }
    hierarchy_data = name_hierarchy_map.get(dataset_name, None)
    hierarchy_descriptions = name_descriptions_map.get(dataset_name, None)
    if hierarchy_data is None or (
        hierarchy_descriptions is None and dataset_name != "wos"
    ):
        raise ValueError(
            f"Dataset {dataset_name} not supported or hierarchy data not found."
        )

    dataset_name = "_" + dataset_name
    hierarchies_folder_path = os.path.join(
        data_files_folder_path,
        "hierarchies",
    )
    if not os.path.exists(hierarchies_folder_path):
        os.makedirs(hierarchies_folder_path)
    file_path = os.path.join(
        data_files_folder_path,
        hierarchies_folder_path,
        f"{dataset_name}_hierarchy_structure.csv",
    )
    if not os.path.exists(file_path):
        new_file_rows = []
        new_file_columns = [
            "label",
            "original_label",
            "description",
        ]
        parent_children_map: dict[str, list[str]] = hierarchy_data["parent_childs"]
        inverse_map: dict[str, str] = {}
        for parent, children in parent_children_map.items():
            for child in children:
                inverse_map[child] = parent
        parent_children_map = {}
        label_to_digit_map: dict[str, str] = {}
        # For each level index from 1 to len(hierarchy_data) - 1
        for i in range(1, len(hierarchy_data) - 1):
            current_level_labels = hierarchy_data[f"level_{i}"]
            roots = []
            for label in current_level_labels:
                # Get the parent label from the inverse map
                parent_label = inverse_map.get(label, "ROOT")
                parent_label_digits = label_to_digit_map.get(parent_label, "")
                # If the current label is a root label, set the available digits to the alphabet
                if parent_label == "ROOT":
                    new_digits = [chr(j) for j in range(ord("A"), ord("Z") + 1)][
                        len(roots) :
                    ][0]
                    roots.append(label)
                else:
                    current_parent_children_len = len(
                        parent_children_map.get(
                            parent_label,
                            [],
                        )
                    )
                    new_digits = f"{current_parent_children_len:02d}"

                new_label_digits = parent_label_digits + new_digits
                label_to_digit_map[label] = new_label_digits
                original_description = (
                    label
                    if hierarchy_descriptions is None
                    else hierarchy_descriptions.get(label, "Not given")
                )
                new_file_rows.append(
                    {
                        "label": new_label_digits,
                        "original_label": label,
                        "description": original_description,
                        "parent_label": parent_label,
                    }
                )

                # Update the parent_children_map
                if parent_label is not None and parent_label not in parent_children_map:
                    parent_children_map[parent_label] = []
                parent_children_map[parent_label].append(label)

        # Write the new file as a CSV file
        new_csv_df = pd.DataFrame(new_file_rows, columns=new_file_columns)
        new_csv_df.to_csv(file_path, index=False)
    else:
        new_csv_df = pd.read_csv(file_path)

    return new_csv_df
