# Проект 7. Ford vs Ferrari: определяем модель авто по фото 
## Юнит 8. Нейронные сети  
### skillfactory_rds  
![https://img.shields.io/badge/Python-3.7.4-blue](https://img.shields.io/badge/Python-3.7.4-blue)

## Оглавление  
[1. Описание модуля](https://github.com/StanislavNevezhin/skillfactory_rds/tree/master/module_8/README.md#Описание-модуля)  
[2. Какой кейс решаем?](https://github.com/StanislavNevezhin/skillfactory_rds/tree/master/module_8/README.md#Какой-кейс-решаем)  
[3. Этапы работы над проектом](https://github.com/StanislavNevezhin/skillfactory_rds/tree/master/module_8/README.md#Этапы-работы-над-проектом)  
[4. Результат](https://github.com/StanislavNevezhin/skillfactory_rds/tree/master/module_8/README.md#Результат) 

### Описание модуля  
Вы работаете в компании, занимающейся продажей автомобилей с пробегом.
Вам нужно построить классификатор изображений для определения модели автомобилей по их фотографиям.   

***Чем мы будем заниматься?***  
- Построим свой классификатор изображений.
- Применим различные методы предобработки изображений.
- Задействуем сразу несколько методов обучения (finetuning, transfer learning и так далее).
- Научимся использовать предобученные модели для решения своих задач.
- Найдем и используем в работе State of the Art (SOTA)-модели.
- Научимся выводить модель DL в продакшн.
:arrow_up:[к оглавлению](https://github.com/StanislavNevezhin/skillfactory_rds/tree/master/module_8/README.md#Оглавление)

### Какой кейс решаем?
Необходимо построить DL-модель для классификации изображений, а также обернуть модель в сервис на Flask для того, чтобы на практике отследить особенности внедрения DL-моделей в продакшн.
В качестве данных в работе над проектом предусмотрено два файла: архивы train.zip и test.zip с фотографиями автомобилей.

**Метрика качества**
Победитель определяется по наибольшему значению accuracy
:arrow_up:[к оглавлению](https://github.com/StanislavNevezhin/skillfactory_rds/tree/master/module_8/README.md#Оглавление)

### Этапы работы над проектом  
1. Анализ датасета EDA.
Проведённый разведовательный анализ позволил подробно ознакомиться с датасетом, увидеть примеры изображений и оценить их размеры для того, чтобы понимать, как их лучше обрабатывать и сжимать при построении модели.
Создан словарь с некорректными изображениями в обучающей выборке, эти изображения были удалены из набора данных.

2. Аугментация данных.
Опробованы различные аргументы Keras ImageDataGenerator для аугментации данных, поскольку работа осуществлялась с небольшим датасетом.

3. Построение модели.
В качестве базовой модели использована SOTA архитектура сети EfficientNetB7, предобученной на ImageNet, добавлена Batch Normalization и построена новая архитектура «головы» (Custom Head) сети.
В процессе обучения модели использовались техники transfer learning и finetuning с постепенной разморозкой весов сети EfficientNetB7 и изменением размера изображений, батча.
Был осуществлен подбор LR, optimizer, loss, количества эпох для обучения, в том числе, использовались другие функции callback в Keras (ModelCheckpoint с сохранением модели в случае улучшения значения val_accuracy, ReduceLROnPlateau с динамическим изменением скорости обучения модели, в случае прекращения снижения val_loss). 

4. Предсказание на тестовых данных.
Была использована техника TTA (Test Time Augmentation) и продвинутые библиотеки аугментации изображений (такие как, albumentations).

3. Вывод модели в Production.	
На основе Flask и Ngrok разработан сервис, позволяющий пользователю в рамках ноутбука Google Colab загружать обученую модель с Google Drive, загружать изображение c локальной машины и предварительно обрабатывать его перед передачей в модель и определять модель автомобиля по загруженному изображению.
:arrow_up:[к оглавлению](https://github.com/StanislavNevezhin/skillfactory_rds/tree/master/module_8/README.md#Оглавление)

### Результат  
- score accuracy = 0.97588 
- 10 место из 133 участников  
:arrow_up:[к оглавлению](https://github.com/StanislavNevezhin/skillfactory_rds/tree/master/module_8/README.md#Оглавление)