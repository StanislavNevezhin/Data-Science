# Проект 4. Компьютер говорит «Нет»
## Юнит 4. Основные алгоритмы машинного обучения. Часть I   
### skillfactory_rds  
![https://img.shields.io/badge/Python-3.7.4-blue](https://img.shields.io/badge/Python-3.7.4-blue)

## Оглавление  
[1. Описание модуля](https://github.com/StanislavNevezhin/skillfactory_rds/tree/master/module_4/README.md#Описание-модуля)  
[2. Какой кейс решаем?](https://github.com/StanislavNevezhin/skillfactory_rds/tree/master/module_4/README.md#Какой-кейс-решаем?)  
[3. Краткая информация о данных](https://github.com/StanislavNevezhin/skillfactory_rds/tree/master/module_4/README.md#Краткая-информация-о-данных)  
[3. Этапы работы над проектом](https://github.com/StanislavNevezhin/skillfactory_rds/tree/master/module_4/README.md#Этапы-работы-над-проектом)  
[4. Результат](https://github.com/StanislavNevezhin/skillfactory_rds/tree/master/module_4/README.md#Результат)  

### Описание модуля  
В этом модуле вас ждёт путь от стажёра отдела аналитики регионального банка до … Будем считать, что почти до начальника отдела!  

***Чем мы будем заниматься?***  
- Напишем скоринговую модель предсказания дефолта клиентов банка.  
- Поучаствуем в командном хакатоне.  
:arrow_up:[к оглавлению](https://github.com/StanislavNevezhin/skillfactory_rds/tree/master/module_4/README.md#Оглавление)

### Какой кейс решаем?
Вы работаете стажёром в отделении регионального банка. Вашей задачей будет построить скоринг модель для вторичных клиентов банка, которая бы предсказывала вероятность дефолта клиента. Для этого нужно будет определить значимые параметры заемщика за 48 часов в формате хакатона.  

**Условия соревнования:**  
Тестовая выборка представлена в ЛидерБорде целиком.  
Поэтому лучшие и победные решения буду проверяться на их "адекватность" (чтоб не было подгонки под тестовую выборку).  
Разрешено использовать любые ML алгоритмы и библиотеки (кроме DL).  
Делаем реальный ML продукт, который потом сможет нормально работать на новых данных.  

**Метрика качества**
Результаты оцениваются по площади под кривой ROC AUC
:arrow_up:[к оглавлению](https://github.com/StanislavNevezhin/skillfactory_rds/tree/master/module_4/README.md#Оглавление)

### Краткая информация о данных
*Определение.* Кредитная история — это карточка заёмщика, в которую записываются все операции с кредитами: какой банк выдавал, сколько есть долгов и вовремя ли платит гражданин. Основание: Федеральный закон «О кредитных историях».  

Датасет содержит информацию о заемщиках, которые уже брали кредиты (повторных клиентов).  
Вам предоставлена информация из анкетных данных заемщиков и факт наличия дефолта.  
Описание полей датасета:  
- client_id	- идентификатор клиента  
- education	- уровень образования (ACD - academy - академическое, PGR - post-graduate - магистратура, GRD - graduate - бакалавриат, SCH - school - среднее)  
- sex	- пол заёмщика  
- age	- возраст заёмщика  
- car	- флаг наличия автомобиля  
- car_type	- флаг автомобиля-иномарки  
- decline_app_cnt	- количество отказанных прошлых заявок  
- good_work	- флаг наличия «хорошей» работы  
- bki_request_cnt	- количество запросов в БКИ  
- home_address	- категоризатор домашнего адреса  
- work_address	- категоризатор рабочего адреса  
- income	- доход заёмщика  
- foreign_passport	- наличие загранпаспорта  
- sna - связь заемщика с клиентами банка  
- first_time - давность наличия информации о заемщике  
- score_bki - скоринговый балл по данным из БКИ  
- region_rating - рейтинг региона  
- app_date - дата подачи заявки  
- default	- наличие дефолта  
:arrow_up:[к оглавлению](https://github.com/StanislavNevezhin/skillfactory_rds/tree/master/module_4/README.md#Оглавление)

### Этапы работы над проектом  
Анализ по переменным.

В ходе анализ строились гистограммы и графики плотности распределения переменных для двух типов заемщиков (дефолтных и недефолтных).

Для выявлкения выбросов в переменных использовались инструменты пакета scikit_posthocs (Simple test based on interquartile range (IQR), Grubbs test, Tietjen-Moore test, Generalized Extreme Studentized Deviate test (ESD test)).

Найдено болшое количество выбросов по переменным decline_app_cnt, score_bki, bki_request_cnt, income, для борьбы с ними были использованы специальные инструменты (винсоризация, тримминг, и т.д. см. https://docs.scipy.org/doc/scipy-0.14.0/reference/stats.mstats.html), а также преобразования, стабилизирующие дисперсию (степенные преобразования, например, логарифмирование, трансформация Бокса-Кокса и т.д.).

Для заполнения пропусков в данных использовались инструменты пакета sklearn.impute (SimpleImputer, KNNImputer).

Построение моделей.

На этом этапе обучались сразу несколько моделей, из которых по метрике ROC-AUC выбирались лучшие для поиска оптимальных параметров моделей. Оценивались следующие модели:
- KNeighborsClassifier;
- GaussianNB;
- LogisticRegression;
- RandomForestClassifier;
- GradientBoostingClassifier;
- XGBClassifier;
- HistGradientBoostingClassifier;
- AdaBoostClassifier;
- Ансамбль AdaBoostClassifier (base_estimator=LogisticRegression);
- LGBMClassifier;
- LogitBoost.

Оценка качества для лучших моделей.

Для визуализации матрицы ошибок, ROC кривой, Precision-Recall кривой, использовались инструменты пакета scikit-plot.

Поиск оптимальных параметров моделей.

Для лучших моделей осуществлялись поиск оптимальных гиперпараметров по сетке и оценка метрики ROC-AUC после оптимизации:
- LogisticRegression - penalty, solver, class_weight;
- GradientBoostingClassifier - learning_rate, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, subsample;
- Ансамбль AdaBoostClassifier + LogisticRegression - learning_rate,n_estimators, algorithm.

Отбор признаков.

Для модели GradientBoostingClassifier осуществлен отбор признаков с использованием инструментов пакета mlxtend.feature selection, в частности Sequential Forward Selection (SFS).

Тестирование альтернативных моделей.

В качестве альтернативной модели использовали классификатор от Яндекса CatBoostClassifier. Модель показала существенно лучший результат, чем Gradient Boosting Scikit-learn, ее и отправили в итоговый Submission.
:arrow_up:[к оглавлению](https://github.com/StanislavNevezhin/skillfactory_rds/tree/master/module_4/README.md#Оглавление)

### Результат  
- score ROC-AUC = 0.74005  
- на 15.09.2020 2 место из 26 участников (Тор 5%)   
:arrow_up:[к оглавлению](https://github.com/StanislavNevezhin/skillfactory_rds/tree/master/module_4/README.md#Оглавление)
