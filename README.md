# Step CV-Pipeline: data_prep

This CV-Pipeline component is designed for processing the dataset: checking and cleaning the dataset, transforming the markup, dividing dataset samples into train, valid and test, reviewing and processing data.

Input data for step CV-Pipeline: data_prep
- **coco_datasets_images**     
images of the downloaded dataset
- **coco_datasets_annotations**    
annotation files of the downloaded dataset

The output of this step CV-Pipeline is
- **coco_train_dataset**     
training dataset
- **coco_eval_dataset**    
validation dataset
- **coco_test_dataset**    
test dataset

## How to run a step CV-Pipeline: data_prep

### Create a directory for the project (or use an existing one)
```
mkdir obj_detect_rest
cd obj_detect_rest
```  

### clone the repository: data_prep
```
git clone --recurse-submodules https://github.com/4-DS/obj_detect_rest-data_prep.git {dir_for_data_prep}
cd {dir_for_data_prep}
```  

### run step CV-Pipeline:data_prep
```
python step.dev.py
```  
or
```
step.prod.py
``` 