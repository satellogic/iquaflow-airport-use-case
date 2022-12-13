import os
import yaml
import cv2
import random
import shutil
import glob

import numpy as np

from typing import Any, Dict, List, Optional, Tuple

from iquaflow.metrics import Metric
from iquaflow.datasets import DSModifier, DSWrapper,DSModifier_jpg
from iquaflow.experiments import ExperimentInfo, ExperimentSetup



class DSModifierJPG(DSModifier):
    """
    Class derived from DSModifier that modifies a dataset iterating its folder. Base class for single-file modifiers.
    Args:
        ds_modifer: Composed modifier child
    Attributes:
        name: Name of the modifier
        ds_modifer: Composed modifier child
        params: Contains metainfomation of the modifier
    """

    def __init__(
        self,
        ds_modifier: Optional[DSModifier] = None,
        params: Dict[str, Any] = {"quality": 65},
    ):
        self.name = f"jpg{params['quality']}_modifier"
        self.params: Dict[str, Any] = params
        self.ds_modifier = ds_modifier
        self.params.update({"modifier": "{}".format(self._get_name())})

    def _ds_input_modification(self, data_input: str, mod_path: str) -> str:
        """Modify images
        Iterates the data_input path loading images, processing with _mod_img(), and saving to mod_path
        Args
            data_input: Path of the original folder containing images
            mod_path: Path to the new dataset
        Returns:
            Name of the new folder containign the images
        """
        input_name = os.path.basename(data_input)
        dst = os.path.join(mod_path, input_name)
        
        if os.path.isfile(dst):
            return input_name
        
        os.makedirs(dst, exist_ok=True)
        
        for data_file in os.listdir(data_input):
            
            file_path = os.path.join(data_input, data_file)
            
            if os.path.isdir(file_path):
                continue
            
            loaded = cv2.imread(file_path, -1)
            
            assert loaded.ndim == 2 or loaded.ndim == 3, (
                "(load_img): File " + file_path + " not valid image"
            )
            
            imgp = self._mod_img(loaded)
            
            cv2.imwrite(os.path.join(dst, data_file), imgp)
            
        return input_name

    def _mod_img(self, img: np.array) -> np.array:
        par = [cv2.IMWRITE_JPEG_QUALITY, self.params["quality"]]
        retval, tmpenc = cv2.imencode(".jpg", img, par)
        rec_img = cv2.imdecode(tmpenc, -1)
        return rec_img


def gen_dataset_partitions(
    seed,
    dsdir = 'datasets/alldata',
    dst_parent_dir = 'datasets',
    ratio_train_val=0.2
):
    """
    This function will generate 2 datasets. One for the partition train and the other for validation.
    
    Arguments:
    
        seed (int): the random seed to be used for the shuffleing of elements in the dataset before partitions.
        dsdir (str): the path to the complete original dataset without partitions
        ratio_train_val (int): the ratio of validation elements with respect to the total (over 1)
        
    Returns:
        
        (List[iquaflow.DSWrapper]): the iquaflow dataset wrappers for train and validation
    """
    src_label_base_fn_lst = os.listdir(os.path.join(dsdir,'labels'))
    
    random.seed(seed)
    random.shuffle(src_label_base_fn_lst)
    
    idx_split = int(len(src_label_base_fn_lst)*ratio_train_val)
    
    for partition in ['train', 'val']:
        for subfold in ['images', 'labels']:
            dst_folder = os.path.join(dst_parent_dir, f'{partition}{seed}', f'{subfold}')
            os.makedirs(dst_folder, exist_ok=True)
    
    for enu, src_label_base_fn in enumerate(src_label_base_fn_lst):
        
        src_label_fn = os.path.join(
            dsdir, 'labels',
            src_label_base_fn
        )
        
        src_img_fn = os.path.join(
            dsdir, 'images',
            src_label_base_fn.split('.')[0]+'.tif'
        )
            
        partition = ('val' if enu < idx_split else 'train')

        dst_img_fn = os.path.join(
            dst_parent_dir,
            f'{partition}{seed}',
            'images',
            os.path.basename(src_img_fn)
        )

        dst_label_fn = os.path.join(
            dst_parent_dir,
            f'{partition}{seed}',
            'labels',
            src_label_base_fn
        )
        
        shutil.copyfile(src_img_fn, dst_img_fn)
        shutil.copyfile(src_label_fn, dst_label_fn)
        
    dsdict =  {
        partition: {
            'data_path': os.path.join(
                dst_parent_dir,
                f'{partition}{seed}'
            ),
            'mask_annotations_dir': os.path.join(
                dst_parent_dir,
                f'{partition}{seed}',
                'labels'
            )
        } for partition in ['train', 'val']
    }

    for partition in ['train', 'val']:
        annotsfn = os.path.join(
            dsdict[partition]['data_path'],
            'annotations.json'
        )
        os.system(f'touch {annotsfn}')
    
    #DS wrapper is the class that encapsulate a dataset
    ref_dsw_train, ref_dsw_val = [
        DSWrapper(
            data_path=dsdict[f'{partition}']['data_path'],
            mask_annotations_dir=dsdict[f'{partition}']['mask_annotations_dir']
        )
        for partition in ['train', 'val']
    ]
    
    return ref_dsw_train, ref_dsw_val