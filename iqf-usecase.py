import os
import random
import shutil

from iquaflow.datasets import DSModifier, DSWrapper,DSModifier_jpg
from iquaflow.experiments import ExperimentInfo, ExperimentSetup
from iquaflow.experiments.experiment_visual import ExperimentVisual
from iquaflow.experiments.task_execution import PythonScriptTaskExecution
from iquaflow.metrics import BBDetectionMetrics

from custom_iqf import DSModifierJPG, gen_dataset_partitions


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
