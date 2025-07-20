import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st


data = pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\EMAILSPAMPROJECTML\spam.csv")


#print(data.head())  prints first and last five rows

print(data.shape)
# cleaning the data
data.drop_duplicates(inplace=True)

data['Category'] = data['Category'].replace(['ham','spam'],['Not Spam','Spam'])

#print(data.isnull().sum())

mess = data['Message']#input message dataset
cat = data['Category']#output message dataset

(mess_train,mess_test,cat_train,cat_test) = train_test_split(mess,cat,test_size=0.2)

#CountVectorizer now it is used to convert text data into numarical data


cv = CountVectorizer(stop_words='english')#not giving much preference to words like is the and ...
features = cv.fit_transform(mess_train)

#CREATING THE MODEL


model = MultinomialNB()

model.fit(features, cat_train)

#testing our model

features_test = cv.transform(mess_test)
#testing accuracy
#print(model.score(features_test,cat_test))

# predict data
def predict(message):
    input_message = cv.transform([message]).toarray()
    result = model.predict(input_message)
    return result[0]


st.header('SPAM DETECTION')

input_mess= st.text_input('Enter Message Here')


if st.button('Validate'):
    output = predict(input_mess)
    st.success(f"The message is: {output}")


#MLmodel SPAM DETECTION
#  TOOLS   1) PANDAS (handles data) 2) SCIKIT_LEARN (creates model , collection of algo)  3) STREAMLIT (python library used to web application)


#for running the code (streamlit run SpamDetection.py)























