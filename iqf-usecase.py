import os
import random
import shutil

from iquaflow.datasets import DSModifier, DSWrapper,DSModifier_jpg
from iquaflow.experiments import ExperimentInfo, ExperimentSetup
from iquaflow.experiments.experiment_visual import ExperimentVisual
from iquaflow.experiments.task_execution import PythonScriptTaskExecution
from iquaflow.metrics import BBDetectionMetrics


def gen_dataset_partitions(
    seed,
    dsdir = 'datasets/alldata',
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
            dst_folder = os.path.join(os.path.dirname(dsdir), f'{partition}{seed}', f'{subfold}')
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
            os.path.dirname(dsdir),
            f'{partition}{seed}',
            'images',
            os.path.basename(src_img_fn)
        )

        dst_label_fn = os.path.join(
            os.path.dirname(dsdir),
            f'{partition}{seed}',
            'labels',
            src_label_base_fn
        )
        
        shutil.copyfile(src_img_fn, dst_img_fn)
        shutil.copyfile(src_label_fn, dst_label_fn)
        
    dsdict =  {
        partition: {
            'data_path': os.path.join(
                os.path.dirname(dsdir),
                f'{partition}{seed}'
            ),
            'mask_annotations_dir': os.path.join(
                os.path.dirname(dsdir),
                f'{partition}{seed}',
                'labels'
            )
        } for partition in ['train', 'val']
    }
    
    #DS wrapper is the class that encapsulate a dataset
    ref_dsw_train, ref_dsw_val = [
        DSWrapper(
            data_path=dsdict[f'{partition}']['data_path'],
            mask_annotations_dir=dsdict[f'{partition}']['mask_annotations_dir']
        )
        for partition in ['train', 'val']
    ]
    
    return ref_dsw_train, ref_dsw_val


def main():

    #Define name of IQF experiment
    experiment_name = "airportV2"

    #List of modifications that will be applied to the original dataset:
    ds_modifiers_list = [
        DSModifier_jpg(params={'quality': i})
        for i in [10,20, 30,50,70,90]
    ]

    # Task execution executes the training loop
    task = PythonScriptTaskExecution(model_script_path = 'train.py')

    for random_seed in [
        47625, 
        29884, 
        51915, 
        89294, 
        71825, 
        57277, 
        39673, 
        80043, 
        14516, 
        90692
    ]:

        ref_dsw_train, ref_dsw_val = gen_dataset_partitions(
            random_seed,
            dsdir = 'datasets/alldata',
            ratio_train_val=0.2
        )

        #Experiment definition, pass as arguments all the components defined beforehand
        experiment = ExperimentSetup(
            experiment_name   = experiment_name,
            task_instance     = task,
            ref_dsw_train     = ref_dsw_train,
            ref_dsw_val       = ref_dsw_val,
            ds_modifiers_list = ds_modifiers_list,
            repetitions=1,
            extra_train_params={"weights":[
                'yolov5n.pt',
                'yolov5s.pt'
            ]}
        )

        #Execute the experiment
        experiment.execute()


    # ExperimentInfo is used to retrieve all the information of the whole experiment. 
    # It contains built in operations but also it can be used to retrieve raw data for futher analysis

    experiment_info = ExperimentInfo(experiment_name)


if __name__ == "__main__":
    
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    
    os.chdir('yolov5')
    
    main()
