![interface data_prep](./imgs/data_prep_inteface.drawio.png)


# Step CV-Pipeline: data_prep

Данная компонент CV-Pipeline предназначен для обработку датасета: проверка и чистка датасета, преобразования разметки, разделение на train, valid и test выборки датасетов, обзор и обработка данных.

Создается на основе [шаблона](https://github.com/4-DS/step_template).
Чтобы не забывать про обязательные ячейки в каждом ноутбуке, проще всего создавать новые ноутбуки просто копированием [`substep_full.ipynb`](https://github.com/4-DS/step_template/blob/main/substep_full.ipynb) из стандартного [шаблона](https://github.com/4-DS/step_template) компоненты.

Конечным выходом работы данного step CV-Pipeline является
- **train_data**     
изображения тренировочного датасета (сохранен как spark parquets)
- **eval_data**    
изображения валидационного датасета (сохранен как spark parquets)
- **test_data**    
изображения тестового датасета (сохранен как spark parquets)
- **train_val_config**    
файлы аннотации для тренировочного и валидационного датасета и необходимый конфигурационный файл для последующей тренировки
- **test_config**    
файлы аннотации для тестового датасета и необходимый конфигурационный файл для запуска тестирования

## Add sinara

### clone repository 
```
git clone https://gitlab.com/yolox_mmdet/data_prep.git
cd data_load
```  

### add sinara module  
```
git submodule add https://github.com/4-DS/sinara.git sinara
```  

### init DSML module  
```
git submodule init
```

### update to latest DSML module
```
git submodule update --remote --merge
```