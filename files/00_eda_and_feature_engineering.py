


import pandas as pd
#import seaborn as sns
import pickle
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

patientdata = pd.read_csv('diabetic_data.csv')

patientdata.head()

patientdata.shape

patientdata.describe()

patientdata.info()

"""#### Age and weight should be numeric, not object - weight and age are both given as ranges rather than a discrete value
#### Also No and Yes should be changed to 0 and 1
"""

patientdata.weight.unique()

patientdata.head()

"""#### Replacing the codes with their actual meanings:"""

patientdata.admission_type_id.replace(
    list(range(1, 9)), ['Emergency',
                        'Urgent',
                        'Elective',
                        'Newborn',
                        'Not Available',
                        'NULL',
                        'Trauma Center',
                        'Not Mapped'], inplace=True)
# patientdata.admission_type_id.head()

id_list = ['Discharged to home',
           'Discharged/transferred to another short term hospital',
           'Discharged/transferred to SNF',
           'Discharged/transferred to ICF',
           'Discharged/transferred to another type of inpatient care institution',
           'Discharged/transferred to home with home health service',
           'Left AMA',
           'Discharged/transferred to home under care of Home IV provider',
           'Admitted as an inpatient to this hospital',
           'Neonate discharged to another hospital for neonatal aftercare',
           'Expired',
           'Still patient or expected to return for outpatient services',
           'Hospice / home',
           'Hospice / medical facility',
           'Discharged/transferred within this institution to Medicare approved swing bed',
           'Discharged/transferred/referred another institution for outpatient services',
           'Discharged/transferred/referred to this institution for outpatient services',
           'NULL',
           'Expired at home. Medicaid only, hospice.',
           'Expired in a medical facility. Medicaid only, hospice.',
           'Expired, place unknown. Medicaid only, hospice.',
           'Discharged/transferred to another rehab fac including rehab units of a hospital .',
           'Discharged/transferred to a long term care hospital.',
           'Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare.',
           'Not Mapped',
           'Unknown/Invalid',
           'Discharged/transferred to another Type of Health Care Institution not Defined Elsewhere',
           'Discharged/transferred to a federal health care facility.',
           'Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital',
           'Discharged/transferred to a Critical Access Hospital (CAH).']

patientdata.discharge_disposition_id.replace(
    list(range(1, len(id_list)+1)), id_list, inplace=True)
patientdata.discharge_disposition_id.head()

id_list = ['Physician Referral',
           'Clinic Referral',
           'HMO Referral',
           'Transfer from a hospital',
           'Transfer from a Skilled Nursing Facility (SNF)',
           'Transfer from another health care facility',
           'Emergency Room',
           'Court/Law Enforcement',
           'Not Available',
           'Transfer from critial access hospital',
           'Normal Delivery',
           'Premature Delivery',
           'Sick Baby',
           'Extramural Birth',
           'Not Available',
           'NULL',
           'Transfer From Another Home Health Agency',
           'Readmission to Same Home Health Agency',
           'Not Mapped',
           'Unknown/Invalid',
           'Transfer from hospital inpt/same fac reslt in a sep claim',
           'Born inside this hospital',
           'Born outside this hospital',
           'Transfer from Ambulatory Surgery Center',
           'Transfer from Hospice']

patientdata.admission_source_id.replace(
    list(range(1, len(id_list)+1)), id_list, inplace=True)
patientdata.admission_source_id.head()

patientdata.head()

# patientdata.drop(patientdata['discharge_disposition_id'].str.contains("Expired"),inplace=True)
patientdata = patientdata[patientdata.discharge_disposition_id.str.contains(
    "Expired") == False]

patientdata.shape

"""#### ICD9 codes from here:
    http://www.icd9data.com/
"""

numeric_code_ranges = [(1, 139),
                       (140, 239),
                       (240, 279),
                       (280, 289),
                       (290, 319),
                       (320, 389),
                       (390, 459),
                       (460, 519),
                       (520, 579),
                       (580, 629),
                       (630, 677),
                       (680, 709),
                       (710, 739),
                       (740, 759),
                       (760, 779),
                       (780, 799),
                       (800, 999)]

numeric_code_ranges

ICD9_diagnosis_groups = ['Infectious And Parasitic Diseases',
                         'Neoplasms',
                         'Endocrine, Nutritional And Metabolic Diseases, And Immunity Disorders',
                         'Diseases Of The Blood And Blood-Forming Organs',
                         'Mental Disorders',
                         'Diseases Of The Nervous System And Sense Organs',
                         'Diseases Of The Circulatory System',
                         'Diseases Of The Respiratory System',
                         'Diseases Of The Digestive System',
                         'Diseases Of The Genitourinary System',
                         'Complications Of Pregnancy, Childbirth, And The Puerperium',
                         'Diseases Of The Skin And Subcutaneous Tissue',
                         'Diseases Of The Musculoskeletal System And Connective Tissue',
                         'Congential Anomalies',
                         'Certain Conditions Originating In The Perinatal Period',
                         'Symptoms, Signs, And Ill-Defined Conditions',
                         'Injury And Poisoning']

ICD9_diagnosis_groups

codes = zip(numeric_code_ranges, ICD9_diagnosis_groups)
code_set = set(codes)

code_set

patientdataICD9 = patientdata.copy()

for num_range, diagnosis in code_set:
    # print(num_range)
    oldlist = range(num_range[0], num_range[1]+1)
    oldlist = [str(x) for x in oldlist]
    newlist = [diagnosis] * len(oldlist)
    for curr_col in ['diag_1', 'diag_2', 'diag_3']:
        patientdataICD9[curr_col].replace(oldlist, newlist, inplace=True)

for curr_col in ['diag_1', 'diag_2', 'diag_3']:
    patientdataICD9[curr_col].replace(oldlist, newlist, inplace=True)
    patientdataICD9.loc[patientdataICD9[curr_col].str.contains(
        'V'), curr_col] = 'Supplementary Classification Of Factors Influencing Health Status And Contact With Health Services'

for curr_col in ['diag_1', 'diag_2', 'diag_3']:
    patientdataICD9[curr_col].replace(oldlist, newlist, inplace=True)
    patientdataICD9.loc[patientdataICD9[curr_col].str.contains(
        'E'), curr_col] = 'Supplementary Classification Of External Causes Of Injury And Poisoning'

for curr_col in ['diag_1', 'diag_2', 'diag_3']:
    patientdataICD9[curr_col].replace(oldlist, newlist, inplace=True)
    patientdataICD9.loc[patientdataICD9[curr_col].str.contains(
        '250'), curr_col] = 'Diabetes mellitus'

print(patientdataICD9['diag_1'].unique())
print(len(patientdataICD9['diag_1'].unique()))

y = patientdata.readmitted

y.head()

x = pd.get_dummies(patientdataICD9.drop(columns=['readmitted', 'encounter_id', 'patient_nbr']))

"""### Class distribution:
The classes are imbalanced, so I'll need to keep that in mind going forward. I'll need to stratify and/or use SMOTE, etc.
"""

#sns.countplot(y)

"""#### Logistic regression, KNN, and SVM all need to be scaled. 

#### I'll use standard scaler (transform to have mean of 0, std of 1).
"""

numeric_columns = list(x.select_dtypes("int64").columns)
numeric_columns

scaler = preprocessing.StandardScaler()

# saving this scaled version of x in case I want to use it outside of test-train split somewhere

x_scaled = x.copy()
x_scaled[numeric_columns] = scaler.fit_transform(x_scaled[numeric_columns])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)

x_train_scaled = x_train.copy()
x_test_scaled = x_test.copy()

# fit_transform on train
x_train_scaled[numeric_columns] = scaler.fit_transform(x_train[numeric_columns])

# transform on test
x_test_scaled[numeric_columns] = scaler.transform(x_test[numeric_columns])

x.head()

x.shape

"""#### Using the broad ICD9 codes brought my x columns down to 269 from 2467

#### Now that these look nice, time to pickle my stuff to use in different models:
"""

with open('x_liv.pkl', 'wb') as picklefile:
    pickle.dump(x, picklefile)

with open('y_liv.pkl', 'wb') as picklefile:
    pickle.dump(y, picklefile)

with open('x_train_liv.pkl', 'wb') as picklefile:
    pickle.dump(x_train, picklefile)

with open('x_test_liv.pkl', 'wb') as picklefile:
    pickle.dump(x_test, picklefile)

with open('y_train_liv.pkl', 'wb') as picklefile:
    pickle.dump(y_train, picklefile)

with open('y_test_liv.pkl', 'wb') as picklefile:
    pickle.dump(y_test, picklefile)

with open('x_train_scaled_liv.pkl', 'wb') as picklefile:
    pickle.dump(x_train_scaled, picklefile)

with open('x_test_scaled_liv.pkl', 'wb') as picklefile:
    pickle.dump(x_test_scaled, picklefile)

with open('x_scaled_liv.pkl', 'wb') as picklefile:
    pickle.dump(x_scaled, picklefile)

with open('patientdataICD9_liv.pkl', 'wb') as picklefile:
    pickle.dump(patientdataICD9, picklefile)
