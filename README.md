# Skolkovo 2022. Трек от FriendWork :moyai:

## Задача
### Кейс   
Автоматизировать и оптимизировать подбор кандидатов на вакансию. Подбор кандидатов вручную занимает в среднем 3,5 часа, а также допускает возможность появления ошибок в результате.

### Описание задачи  
В среднем процесс подбора занимает 3,5 часа и имеет большое кол-во допускаемых ошибок. Задача, решаемая участниками, сможет помочь компании  закрыть проблему клиента, автоматизировать и сократить до 50 минут процесс ранжирования, предоставляя готовый релевантный вакансиям результат. 

Задача, закрываемая участниками, даст новые идеи для компании FriendWork, а также решит проблему многих компаний, которые тратят время и деньги на долгий процесс подбора.
На основе данных кандидатов и вакансий на которые они были приняты необходимо обучить модель предсказания соответствия кандидата предлагаемой вакансии.

Исходные данные представляют из себя параметризированный список кандидатов, параметризированный список вакансий, а также отношения кандидата и вакансии - был ли кандидат на нее принят, был ли отказ по позиции и тд.

Параметр соответствия вакансии и кандидата должен быть значением от 0 до 1.

### Бизнес-требования  
Модуль (модель) принимает на вход список кандидатов и вакансию, возвращает список кандидатов отсортированных по метрике (от 0 до 1) соответствия предлагаемой вакансии от максимального соответствия к минимальному.

### Технические требования и ограничения  
1. Время работы ранжирования модели при входных параметрах 1 вакансия, 10 кандидатов - не более 1с

2. Возможность потокового обучения (возможность внедрения в соответствии с методологией CRISP-DM)

3. Использование технологии с возможностью дальнейшего представления функционала как микросервис

4. Веб-интерфейс с возможностью ввести данные по вакансии и кандидату с возможностью последующего получения коэффициента соответствия.

 

### Для оценки необходимо предоставить  
- Исходный код;  
-  Результирующие файлы для оценки (10 по количеству тестовых вакансий);  
- Презентацию;  
- Ссылку на веб-интерфейс.  

### Критерии оценки  
Предоставлены результирующие файлы по 10 вакансиям с ранжированными кандидатами в соответствии с их реальным соответствием вакансиям
Наличие и корректная работа веб-интерфейса для проверки соответствия вакансии и кандидата. Оценивается удобство использования и корректность коэффициента.

## Данные

- [Диск](https://drive.google.com/drive/folders/10F0gjEGF_46Sr1Ck5hvORo1wT3xdKO24?usp=sharing) с исходными данными;  


## Результаты

- [Предикты](https://drive.google.com/drive/folders/10nSRddXjc61tMGTSaeIOIx_VzqU00v6K?usp=sharing) для тестов;  
- [Презентация](https://github.com/PunkButterfly/hackathon-skolkovo2022/blob/main/presentation.pdf).  


## Web-сервис

[Веб-сервис](https://kealfeyne-skolkovo-deploy-info-poggx6.streamlitapp.com/Info)  
 * _Match_ - страница, на которой вводятся данные о кандидате и вакансии. После нажатия кнопки **Predict** выводится score соответствия кандидата вакансии;  
 * _Upload_ _csv_ - страница, на которую загружаются 4 .csv файла, соответствующих тестовым. После нажатия на **Predict** начинает билдиться архив с файлами типа results_job (список ранжированных кандидатов для каждой вакансии из файла jobs).


## Обзор файлов

- [data_analysis](https://github.com/PunkButterfly/hackathon-skolkovo2022/blob/main/data_analysis.ipynb) - первичное исследование данных;  
- [data_preprocessing.ipynb](https://github.com/PunkButterfly/hackathon-skolkovo2022/blob/main/data_preprocessing.ipynb) - обработка сырых данных перед обучением модели и предиктом;  
- [train_notebook.ipynb](https://github.com/PunkButterfly/hackathon-skolkovo2022/blob/main/train_notebook.ipynb) - построение и обучение модели.  
