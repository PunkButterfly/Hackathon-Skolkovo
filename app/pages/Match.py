from transformers import AutoTokenizer, AutoModel
import streamlit as st
import processing as pr
import prediction
from time import time

st.header('Input parameters')
job_parameters = {}
candidate_parameters = {}
degree_parameters = {}
workplaces_parameters = {}

job_column, candidate_column = st.columns(2, gap='large')
# Job parameters
with job_column:
    st.subheader('Vacancy')

    job_parameters['JobId'] = None  # st.number_input('Job ID', step=1)
    status_dict = {'Opened': 1, 'Completed': 2, 'Stopped': 3, 'Canceled': 4, 'Drafted': 5}
    job_parameters['Status'] = status_dict.get(st.selectbox('Job status', status_dict.keys()), 5)
    job_parameters['Name'] = st.text_input('Title')
    job_parameters['Region'] = st.text_input('Region')
    job_parameters['Description'] = st.text_area('Description')
    job_parameters['to_del1'] = None
    job_parameters['to_del2'] = None
    job_parameters['to_del3'] = None

# Candidate parameters
with candidate_column:
    st.subheader('Candidate')

    candidate_parameters['CandidateId'] = 100  # st.number_input('Candidate ID', step=1)
    candidate_parameters['Position'] = st.text_input('Position')
    sex_dict = {'Male': 2, 'Female': 1, 'Undefined': 0}
    candidate_parameters['Sex'] = sex_dict.get(st.selectbox('Sex', sex_dict.keys()), 0)
    candidate_parameters['Citizenship'] = st.text_input('Citizenship')
    candidate_parameters['Age'] = st.number_input('Age', step=1)
    candidate_parameters['Salary'] = st.number_input('Salary', step=1)
    candidate_parameters['Langs'] = st.text_input('Languages')
    candidate_parameters['DriverLicense'] = st.text_input('Driver license')
    candidate_parameters['Subway'] = st.text_input('Subway')
    candidate_parameters['Skills'] = st.text_area('Skills')
    employment_dict = {'Full': 0, 'Part': 1, 'Look-out': 2, 'Change': 3,
                       'Not full': 4, 'Does not matter': 5, 'None': None}
    candidate_parameters['Employment'] = employment_dict.get(
        st.selectbox('Employment', employment_dict.keys()), None)
    schedule_dict = {'Full': 0, 'Flex': 1, 'Project': 2, 'Stage': 3,
                     'Volunteering': 4, 'Does not matter': 5, 'None': None}
    candidate_parameters['Shedule'] = schedule_dict.get(st.selectbox('Schedule', schedule_dict.keys()), None)
    candidate_parameters['CandidateRegion'] = st.text_input('Candidate region')
    # candidate_parameters['DateCreated'] = None #  st.date_input('Date created')
    # candidate_parameters['JobId'] = None #  st.number_input('Vacancy ID', step=1)
    # candidate_parameters['CandidateStatusId'] = None #  st.number_input('Candidate status ID', step=1)
    # candidate_parameters['Status'] = None #  st.text_area('Status description')
# Degree parameters
    st.subheader('Candidate`s degree')

    degree_parameters['CandidateId'] = candidate_parameters['CandidateId']
    degree_parameters['University'] = st.text_input('University')
    degree_parameters['Faculty'] = st.text_input('Faculty')
    degree_parameters['GraduateYear'] = st.number_input('Graduate Year', step=1)

# Workplaces parameters
    st.subheader('Past candidate`s workplace')

    workplaces_parameters['CandidateId'] = candidate_parameters['CandidateId']
    workplaces_parameters['Position'] = st.text_input('Past position')
    workplaces_parameters['FromYear'] = st.number_input('From year', step=1)
    workplaces_parameters['FromMonth'] = st.number_input('From month', step=1)
    workplaces_parameters['ToYear'] = st.number_input('To year', step=1)
    workplaces_parameters['ToMonth'] = st.number_input('To month', step=1)

footer1, footer2, footer3 = st.columns(3, gap='large')
result = None
with footer3:
    bert_tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
    bert_model = AutoModel.from_pretrained("cointegrated/rubert-tiny")
    if st.button('Predict'):
        time_start = time()
        preprocessed_data = pr.preprocess(job_parameters, candidate_parameters,
                                          degree_parameters, workplaces_parameters)
        processed_data = pr.process(preprocessed_data[0], preprocessed_data[1],
                                    preprocessed_data[2], preprocessed_data[3])

        result = prediction.inference('app/fully_model.pt', processed_data.loc[0, :], bert_tokenizer, bert_model)
        st.write(f'Time spent: {time() - time_start}')
st.success(f'Score: {result}')
