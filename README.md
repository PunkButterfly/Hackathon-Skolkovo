# Решение задачи от команды **Punk Butterfly** :moyai:

## Результаты

- [Директория с предиктами для тестов](https://git.codenrock.com/skolkovo-hack-2022/cnrprod-team-24259/friend-work-task/-/tree/main/data/result);  
- [Презентация](https://git.codenrock.com/skolkovo-hack-2022/cnrprod-team-24259/friend-work-task/-/blob/main/%D0%9F%D1%80%D0%B5%D0%B7%D0%B5%D0%BD%D1%82%D0%B0%D1%86%D0%B8%D1%8F_Punk_Butterfly.pdf).  


## Web-сервис
[Веб-сервис](https://kealfeyne-skolkovo-deploy-info-poggx6.streamlitapp.com/Match)  
 * _Match_ - страница, на которой вводятся данные о кандидате и вакансии. После нажатия кнопки **Predict** выводится score соответствия кандидата вакансии;  
 * _Upload_ _csv_ - страница, на которую загружаются 4 .csv файла, соответствующих тестовым. После нажатия на **Predict** начинает билдиться архив с файлами типа results_job(список ранжированных кандидатов для каждой вакансии из файла jobs).
## Обзор файлов

- [Plots.ipynb](https://git.codenrock.com/skolkovo-hack-2022/cnrprod-team-24259/friend-work-task/-/blob/main/Plots.ipynb) - первичное исследование данных;  
- [preprocessing_data.ipynb](https://git.codenrock.com/skolkovo-hack-2022/cnrprod-team-24259/friend-work-task/-/blob/main/preprocessing_data.ipynb) - обработка сырых данных перед обучением модели и предиктом;  
- [train_notebook.ipynb](https://git.codenrock.com/skolkovo-hack-2022/cnrprod-team-24259/friend-work-task/-/blob/main/train_notebook.ipynb) - построение и обучение модели;  
- [preprocess_data.csv](https://git.codenrock.com/skolkovo-hack-2022/cnrprod-team-24259/friend-work-task/-/blob/main/preprocess_data.csv) - обработанные данные для обучения.