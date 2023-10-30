![interface data_prep](./imgs/data_prep_inteface.drawio.png)


# Step CV-Pipeline: data_prep

Данная компонент CV-Pipeline предназначен для обработку датасета: проверка и чистка датасета, преобразования разметки, разделение на train, valid и test выборки датасетов, обзор и обработка данных.

Создается на основе [шаблона](https://github.com/4-DS/step_template).
Чтобы не забывать про обязательные ячейки в каждом ноутбуке, проще всего создавать новые ноутбуки просто копированием [`substep_full.ipynb`](https://github.com/4-DS/step_template/blob/main/substep_full.ipynb) из стандартного [шаблона](https://github.com/4-DS/step_template) компоненты.

Входные данные для step CV-Pipeline: data_prep
- **images**     
изображения скачанного датасета (сохранен как spark parquets)
- **annotations**    
файлы аннотации скачанного датасета (сохранен как spark parquets)

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


## Как запустить шаг CV-Pipeline: model_prep

### Создать директорию для проекта (или использовать уже существующую)
```
mkdir yolox_mmdet
cd yolox_mmdet
```  

### склонировать репозиторий model_prep
```
git clone --recurse-submodules https://gitlab.com/yolox_mmdet/data_prep.git {model_prep}
cd model_prep
```  

### запустить шаг CV-Pipeline:model_prep
```
python step.dev.py
```  
или
```
step.prod.py
``` 