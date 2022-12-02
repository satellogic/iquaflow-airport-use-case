import os

from iq_tool_box.experiments import ExperimentInfo, ExperimentSetup
from iq_tool_box.experiments.task_execution import PythonScriptTaskExecution

#Define name of IQF experiment
experiment_name = "airport"

#DS wrapper is the class that encapsulate a dataset
ref_dsw_train = DSWrapper(data_path="datasets/train")

#DS wrapper is the class that encapsulate a dataset
ref_dsw_val = DSWrapper(data_path="datasets/validation")

#List of modifications that will be applied to the original dataset:
ds_modifiers_list = [
    DSModifier_jpg(params={'quality': i})
    for i in [10,30,50,70,90]
]

# Task execution executes the training loop
task = PythonScriptTaskExecution(model_script_path = 'yolov5/train.py')

#Experiment definition, pass as arguments all the components defined beforehand
experiment = ExperimentSetup(
    experiment_name   = experiment_name,
    task_instance     = task,
    ref_dsw_train     = ref_dsw_train,
    ref_dsw_val       = ref_dsw_val,
    ds_modifiers_list = ds_modifiers_list,
    repetitions=1,
    extra_params: {"weights":['yolov5n.pt', 'yolov5s.pt']}
)

#Execute the experiment
experiment.execute()

# ExperimentInfo is used to retrieve all the information of the whole experiment. 
# It contains built in operations but also it can be used to retrieve raw data for futher analysis

experiment_info = ExperimentInfo(experiment_name)
