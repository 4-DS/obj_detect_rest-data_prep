from pathlib import Path
from tqdm import tqdm
import os.path as osp
import shutil

def get_files(path, extensions, relative=True):

    if type(extensions) is str:
        extensions = [extensions]

    all_files = []
    for ext in extensions:
        if relative:
            all_files.extend(Path(path).rglob(ext))
        else:
            all_files.extend(Path(path).glob(ext))

    for i in range(len(all_files)):
        all_files[i] = str(all_files[i])

    return all_files


def prepare_coco_dataset_images(_coco_data, source_img_folder:str, dest_img_folder: str):
    pack = []
    for img_info in tqdm(_coco_data["images"]):
        source_file_name = osp.join(source_img_folder, img_info["file_name"])
        dest_file_name = osp.join(dest_img_folder, img_info["file_name"])
        shutil.copyfile(source_file_name, dest_file_name)