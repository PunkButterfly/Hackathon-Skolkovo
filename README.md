# Skolkovo 2022. Трек от FriendWork :moyai:

## Задача

[Задача](https://codenrock.com/contests/skolkovo-hack-2022#/info) ранжирования соответствия пула кандидатов предлагаемой вакансии.  


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
