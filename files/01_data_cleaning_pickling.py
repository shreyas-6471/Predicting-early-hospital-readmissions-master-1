
import pandas as pd
import seaborn as sns
import pickle
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier

# eventually relocated to plots.py:
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix

df = pd.read_csv('diabetic_data.csv')

df.head()

"""#### Age and weight should be numeric, not object - weight and age are both given as ranges rather than a discrete value
#### Also No and Yes should be changed to 0 and 1
"""

# Replace age ranges with numerical values:
age_id = {'[0-10)':0, 
        '[10-20)':10, 
        '[20-30)':20, 
        '[30-40)':30, 
        '[40-50)':40, 
        '[50-60)':50,
        '[60-70)':60, 
        '[70-80)':70, 
        '[80-90)':80, 
        '[90-100)':90}
df['age'] = df.age.replace(age_id)

df.weight.unique()

# Replace weight ranges with numerical values:
# These are mostly missing values, so the other idea is we could just code whether missing or not
# We could one-hot encode them instead also, but they're mostly missing
# Assigning the missing value to 0 in this label encoding is a bit flawed, but might work fine
weight_dict = {'?':0, 
               '[75-100)':100, 
               '[50-75)':75, 
               '[0-25)':25, 
               '[100-125)':125, 
               '[25-50)':50,
               '[125-150)':150, 
               '[175-200)':200, 
               '[150-175)':175, 
               '>200':225}
df['weight'] = df.weight.replace(weight_dict)

df.weight.unique()

df.shape

df.describe()

df.info()

df.head()

df.head()

"""## Drop unique identifiers:"""

df = df.drop(['encounter_id','patient_nbr'],axis=1)

df['glyburide-metformin'].unique()

df['insulin'].unique()

df['chlorpropamide'].unique()

df['diag_1'].unique()

df['A1Cresult'].unique()

df.head()

df.select_dtypes(exclude=["number","bool_","object_"]).columns

# Check missing values
df = df.replace("?",np.NaN)
df.isna().mean().round(4) * 100

df = df.replace(np.NaN,"UNK")

df.weight.unique()

"""#### Replacing the codes with their actual meanings:"""

df.admission_type_id.replace(
list(range(1,9)),['Emergency',
'Urgent',
'Elective',
'Newborn',
'Not Available',
'NULL',
'Trauma Center',
'Not Mapped'], inplace=True)
#df.admission_type_id.head()

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

df.discharge_disposition_id.replace(list(range(1,len(id_list)+1)),id_list, inplace=True)
df.discharge_disposition_id.head()

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

df.admission_source_id.replace(list(range(1,len(id_list)+1)),id_list, inplace=True)
df.admission_source_id.head()

df.head()

# Remove patients that have expired; these patients will never be readmitted

#df.drop(df['discharge_disposition_id'].str.contains("Expired"),inplace=True)
df = df[df.discharge_disposition_id.str.contains("Expired") == False]

df.shape

df.loc[df['diag_1'].str.contains('V'), "diag_1"]

df.loc[df['diag_2'].str.contains('V'), "diag_2"]

df.loc[df['diag_3'].str.contains('V'), "diag_3"]

df.loc[df['diag_1'].str.contains('E'), "diag_1"]

df.loc[df['diag_2'].str.contains('E'), "diag_2"]

df.loc[df['diag_3'].str.contains('E'), "diag_3"]

df.loc[df['diag_1'].str.contains('250'), "diag_1"]

"""#### ICD9 codes from here:
    http://www.icd9data.com/
"""

numeric_code_ranges = [(1,139),
(140,239),
(240,279),
(280,289),
(290,319),
(320,389),
(390,459),
(460,519),
(520,579),
(580,629),
(630,677),
(680,709),
(710,739),
(740,759),  
(760,779),  
(780,799),  
(800,999)]

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
codeSet = set(codes)

codeSet

for num_range, diagnosis in codeSet:
    #print(num_range)
    oldlist = range(num_range[0],num_range[1]+1)
    oldlist = [str(x) for x in oldlist]
    newlist = [diagnosis] * len(oldlist)
    for curr_col in ['diag_1', 'diag_2', 'diag_3']:
        df[curr_col].replace(oldlist, newlist, inplace=True)

for curr_col in ['diag_1', 'diag_2', 'diag_3']:
    df[curr_col].replace(oldlist, newlist, inplace=True)
    df.loc[df[curr_col].str.contains('V'), curr_col] = 'Supplementary Classification Of Factors Influencing Health Status And Contact With Health Services'
    df.loc[df[curr_col].str.contains('E'), curr_col] = 'Supplementary Classification Of External Causes Of Injury And Poisoning'
    df.loc[df[curr_col].str.contains('250'), curr_col] = 'Diabetes mellitus'

print(df['diag_1'].unique())
print(len(df['diag_1'].unique()))

y = df['readmitted']
y = y.replace({'<30': 1, '>30': 0, 'NO': 0})
df = df.drop(['readmitted'],axis=1)

y.head()

sns.countplot(y);

"""#### The classes are biased, so I'll need to keep that in mind going forward. I'll need to stratify and/or use SMOTE, etc."""

list(df.select_dtypes(["number","bool_"]).columns)

"""## Save with categoricals for fastai tabular learner"""

with open('y_2.pkl', 'wb') as picklefile:
    pickle.dump(y, picklefile)

with open('x_2_with_categoricals.pkl', 'wb') as picklefile:
    pickle.dump(df, picklefile)

"""## One hot encoding:"""

cat_cols = list(df.select_dtypes(exclude=["number","bool_"]).columns)
cat_cols

df_cat = pd.get_dummies(df[cat_cols],drop_first = True)
df = pd.concat([df.drop(cat_cols,axis=1),df_cat], axis = 1)

"""## Save one-hot encoded:"""

with open('x_2.pkl', 'wb') as picklefile:
    pickle.dump(df, picklefile)

"""## Quick test of model performance with this preprocessing"""

scaler = preprocessing.StandardScaler()

y = y.replace({'<30': 1, '>30': 0, 'NO': 0})
class_names = ["not early readmit", "early readmit"]

# stratified split
X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=42, stratify=y)

# standard scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# smote oversampling
smote = SMOTE()


X_train, y_train = smote.fit_resample(X_train, y_train)

model = ExtraTreesClassifier(n_estimators=100, n_jobs=-1, min_samples_split=25,
                             min_samples_leaf=35, max_features=150)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_pred_proba = model.predict_proba(X_test)

print(classification_report(y_test, y_pred))

def precision_recall_plot(y_test, y_pred_proba):
    p, r, t = precision_recall_curve(y_test, y_pred_proba[:, 1])

    # adding last threshold of '1' to threshold list
    t = np.vstack([t.reshape([-1, 1]), 1])
    
    # boxplot algorithm comparison
    fig = plt.figure(figsize=(10,10))
    fig.suptitle('Precision Recall Curve')
    ax = fig.add_subplot(111)
    plt.plot(t, p, label="precision")
    plt.plot(t, r, label="recall")
#     plt.show()

    return fig

precision_recall_plot(y_test, y_pred_proba)

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=sns.color_palette("Blues")):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Only use the labels that appear in the data
#     classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    ax = sns.heatmap(cm, cmap = cmap, annot=True, xticklabels=classes, yticklabels=classes)

    ax.set(title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the x labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Vertically center y labels
    plt.setp(ax.get_yticklabels(), va="center")
    
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    
    return ax

# Plot non-normalized confusion matrix
ax = plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix, without normalization')

# plt.savefig('confusion_matrix_test.png', bbox_inches="tight")