from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

from detectron2.data import DatasetCatalog, MetadataCatalog

from cubercnn.data.datasets import load_omni3d_json

DATA_PATH = ROOT / "data"


def simple_register(dataset_name, filter_settings, filter_empty=False):
    path_to_json = (DATA_PATH / f"{dataset_name}.json").as_posix()

    DatasetCatalog.register(
        dataset_name,
        lambda: load_omni3d_json(
            path_to_json,
            DATA_PATH,
            dataset_name,
            filter_settings,
            filter_empty=filter_empty,
        ),
    )
    print(f"Registered {dataset_name}")

    MetadataCatalog.get(dataset_name).set(
        json_file=path_to_json, image_root=DATA_PATH, evaluator_type="coco"
    )
