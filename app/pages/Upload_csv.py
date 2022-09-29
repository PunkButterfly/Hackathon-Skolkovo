from transformers import AutoTokenizer, AutoModel
import streamlit as st
import pandas as pd
import processing as pr
import prediction
from zipfile import ZipFile
from io import BytesIO
from time import time
import base64

bert_tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
bert_model = AutoModel.from_pretrained("cointegrated/rubert-tiny")

st.header('You can upload a csv files')
st.subheader('')
job = st.file_uploader('Upload jobs.csv')
candidates = st.file_uploader('Upload candidates.csv')
candidates_education = st.file_uploader('Upload candidates_education.csv')
candidates_workplaces = st.file_uploader('Upload candidates_workplaces.csv')

result = None
if (job is not None) and (candidates is not None) and\
        (candidates_education is not None) and (candidates_workplaces is not None):
    job = pd.read_csv(job, sep=';',
                      names=['JobId', 'Status', 'Name', 'Region', 'Description', 'to_del1', 'to_del2', 'to_del3'])
    candidates = pd.read_csv(candidates, sep=';')
    candidates_education = pd.read_csv(candidates_education, sep=';')
    candidates_workplaces = pd.read_csv(candidates_workplaces, sep=';')

    if st.button('Predict'):
        jobs_results = []
        for job_index in range(0, job.shape[0]):
            time_start = time()
            job_id = job["JobId"].iloc[job_index]

            jobs_results.append(pd.DataFrame([], columns=['CandidateID', 'Score']))
            processed_data = pr.process(pd.DataFrame(job.iloc[[job_index]], columns=job.columns),
                                        candidates, candidates_education, candidates_workplaces)
            for candidate_index in range(0, candidates.shape[0]):

                result = prediction.inference('fully_model.pt', processed_data.loc[candidate_index, :],
                                              bert_tokenizer, bert_model)
                jobs_results[job_index] = jobs_results[job_index].append(pd.DataFrame(
                    [[candidates['CandidateId'][candidate_index], result]], columns=['CandidateID', 'Score']),
                    ignore_index=True)

            st.success(f'Results for Job {job_id} are ready!\n Time spent: {format(time() - time_start, ".2f")}')

        zipObj = ZipFile("results", "w")
        for file_index in range(0, len(jobs_results)):
            job_id = job["JobId"].iloc[file_index]

            jobs_results[file_index].sort_values(['Score'], ascending=False, inplace=True)
            jobs_results[file_index].to_csv(f'result_job_{job_id}.csv', header=False,
                                            sep=';', index=False, encoding='utf-8')
            zipObj.write(f'result_job_{job_id}.csv')

        zipObj.close()

        ZipfileDotZip = 'results'
        with open(ZipfileDotZip, 'rb') as f:
            bytes = f.read()
            b64 = base64.b64encode(bytes).decode()
            href = f"<a href=\"data:file/zip;base64,{b64}\" download='{ZipfileDotZip}.zip'>\
                    Download results\
                </a>"
            st.markdown(href, unsafe_allow_html=True)
