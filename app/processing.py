import pandas as pd
import numpy as np
import streamlit as st
import re


def preprocess(job_dict, candidate_dict, degree_dict, workplaces_dict):
    job_features_df = pd.DataFrame([job_dict], columns=list(job_dict.keys()))
    candidate_features_df = pd.DataFrame([candidate_dict], columns=list(candidate_dict.keys()))
    degree_parameters_df = pd.DataFrame([degree_dict], columns=list(degree_dict.keys()))
    workplaces_dict_df = pd.DataFrame([workplaces_dict], columns=list(workplaces_dict.keys()))

    return job_features_df, candidate_features_df, degree_parameters_df, workplaces_dict_df


def process(test_jobs, test_candidates, test_candidates_education, test_candidates_workplaces):

    def cleanhtml(raw_html):
        if type(raw_html) == type(''):
            return re.sub(CLEANR, '', raw_html)
        return raw_html

    def del_sep(text):
        if type(text) == type(''):
            return ''.join(text.split(','))
        return text

    def del_stick(text):
        if type(text) == type(''):
            return ''.join(text.split('||'))
        return text

    CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});|-|/|;|_|•')
    test_jobs.Description = test_jobs.Description.apply(cleanhtml)

    status_jobs = pd.DataFrame(
        [[1, 'открытая'], [2, 'закрытая'], [3, 'приостановленная'], [4, 'отменена'], [5, 'черновик']],
        columns=['Status', 'name_status'])
    new_data_jobs = test_jobs \
        .merge(status_jobs, on='Status')
    new_data_jobs.index = new_data_jobs.JobId
    job_user_tokens1 = new_data_jobs \
        .drop(columns=['Status', 'JobId']) \
        .fillna('NaN') \
        .astype(str) \
        .add('. ') \
        .sum(axis=1) \
        .reset_index()

    candidates_work_places_concat = test_candidates_workplaces.drop_duplicates(['CandidateId', 'Position'])
    candidates_work_places_concat.Position = candidates_work_places_concat.Position.add(' ')
    candidates_work_places_concat = candidates_work_places_concat.groupby('CandidateId', as_index=False).agg(
        {'Position': 'sum'})

    test_candidates.DriverLicense = test_candidates.DriverLicense.apply(del_sep)

    test_candidates.Skills = test_candidates.Skills.apply(del_stick)

    new_df = test_candidates \
        .merge(candidates_work_places_concat, on='CandidateId', how='left') \
        .merge(test_candidates_education.drop_duplicates('CandidateId'), on='CandidateId', how='left')
    new_df.index = new_df.CandidateId
    df = new_df.drop(columns=['CandidateId', 'Salary', 'Subway',
                              'Age', 'GraduateYear'])
    df.Sex = df.Sex \
        .where(~(df.Sex == 2), other='мужчина') \
        .where(~(df.Sex == 1), other='женщина') \
        .where(~(df.Sex == 0), other='неопределен')
    df = df.fillna('NaN').astype(str).add('. ').sum(axis=1).reset_index()

    job_user_tokens2 = df

    job_user_tokens2 = job_user_tokens2.rename(columns={0: 'Candidate_descr'})
    job_user_tokens1 = job_user_tokens1.rename(columns={0: 'Jobs_descr'})
    job_user_tokens2.drop(columns=['CandidateId'], inplace=True)
    job_user_tokens1.drop(columns=['JobId'], inplace=True)

    test_data_merged = job_user_tokens2.merge(job_user_tokens1, how='cross')

    return test_data_merged
