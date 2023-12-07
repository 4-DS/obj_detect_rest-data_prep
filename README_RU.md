# Step CV-Pipeline: data_prep

Данная компонент CV-Pipeline предназначен для обработку датасета: проверка и чистка датасета, преобразования разметки, разделение на train, valid и test выборки датасетов, обзор и обработка данных.

Входные данные для step CV-Pipeline: data_prep
- **coco_datasets_annotations**     
изображения скачанного датасета
- **coco_datasets_annotations**    
файлы аннотации скачанного датасета

Выходом работы данного step CV-Pipeline является
- **coco_train_dataset**     
тренировочный датасет
- **coco_eval_dataset**    
валидационный датасет
- **coco_test_dataset**    
тестовый датасет

## Как запустить шаг CV-Pipeline: model_prep

### Создать директорию для проекта (или использовать уже существующую)
```
mkdir obj_detect_binary
cd obj_detect_binary
```  

### склонировать репозиторий model_prep
```
git clone --recurse-submodules https://github.com/4-DS/obj_detect_binary-data_prep.git {dir_for_data_prep}
cd {dir_for_data_prep}
```  

### запустить шаг CV-Pipeline:model_prep
```
python step.dev.py
```  
или
```
step.prod.py
``` 