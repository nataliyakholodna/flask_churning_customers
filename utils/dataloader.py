import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class DataLoader(object):

    def fit(self, dataset):
        self.dataset = dataset.copy()

    def load_data(self):

        # drop columns
        drop_columns = ['CLIENTNUM',
                        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2']

        self.dataset = self.dataset.drop(columns=drop_columns)

        le = LabelEncoder()

        le.fit(self.dataset['Gender'])
        self.dataset['Gender'] = le.transform(self.dataset['Gender'])

        le.fit(self.dataset['Card_Category'])
        self.dataset['Card_Category'] = le.transform(self.dataset['Card_Category'])

        le.fit(self.dataset['Income_Category'])
        self.dataset['Income_Category'] = le.transform(self.dataset['Income_Category'])

        le.fit(self.dataset['Education_Level'])
        self.dataset['Education_Level'] = le.transform(self.dataset['Education_Level'])

        le.fit(self.dataset['Marital_Status'])
        self.dataset['Marital_Status'] = le.transform(self.dataset['Marital_Status'])

        '''
        # binary encoding

        gender_flag = {
            'M': 1,
            'F': 0
        }

        self.dataset['Gender'] = self.dataset['Gender'].map(gender_flag)

        # label encoding

        self.dataset['Card_Category'] = pd.Categorical(self.dataset['Card_Category'], categories=['Blue', 'Gold', 'Silver', 'Platinum'])
        self.dataset['Card_Category'] = self.dataset['Card_Category'].cat.codes

        self.dataset['Income_Category'] = pd.Categorical(self.dataset['Income_Category'],
                                                  categories=['Unknown', 'Less than $40K', '$40K - $60K', '$80K - $120K',
                                                              '$60K - $80K','$120K +'])

        self.dataset['Income_Category'] = self.dataset['Income_Category'].cat.codes

        # one-hot

        self.dataset = pd.get_dummies(self.dataset, columns=['Education_Level', 'Marital_Status'])
        self.dataset.drop(columns=['Education_Level_Unknown', 'Marital_Status_Unknown'], inplace=True)
        '''

        return self.dataset



