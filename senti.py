import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
import matplotlib.pyplot as plt
import seaborn as sns

# Function to preprocess the data
@st.cache_data()
def preprocessing_data(data):
    processing_data = []
    for single_data in data:
        if len(single_data.split("\t")) == 2 and single_data.split("\t")[1] != "":
            processing_data.append(single_data.split("\t"))
    return processing_data

# Function to split data into training and evaluation sets
def split_data(data):
    total = len(data)
    training_ratio = 0.75
    training_data = data[:int(total * training_ratio)]
    evaluation_data = data[int(total * training_ratio):]
    return training_data, evaluation_data

# Function to train the model
def training_step(data,vectorizer):
    training_text = [data[0] for data in data]
    training_result = [data[1] for data in data]
    training_text = vectorizer.fit_transform(training_text)
    return BernoulliNB().fit(training_text, training_result)

# Function to analyze text
def analyse_text(classifier, vectorizer, text):
    return text, classifier.predict(vectorizer.transform([text]))

# Function to print result
def print_result(result):
    text, analysis_result = result
    print_text = "Positive" if analysis_result[0] == '1' else "Negative"
    return text, print_text

# Streamlit UI
st.image("Amazon image.png", width=200)
st.title("Amazon Product Review Analyzer")
st.write('\n\n')

# Sidebar navigation
st.markdown(
    f"""
    <style>
    .sidebar .sidebar-content {{
        background-color: #F5F5DC;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
# Sidebar navigation
nav_selection = st.sidebar.radio("Navigation", ["Home", "Analyzer"])

if nav_selection == "Analyzer":
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: #F5F5DC;
            color: black;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    # Load dataset and preprocess it
    root = "Datasets/"
    with open(root + "imdb_labelled.txt", "r") as text_file:
        data = text_file.read().split('\n')

    with open(root + "amazon_cells_labelled.txt", "r") as text_file:
        data += text_file.read().split('\n')

    with open(root + "yelp_labelled.txt", "r") as text_file:
        data += text_file.read().split('\n')

    all_data = preprocessing_data(data)

    

    # Training and evaluation
    training_data, evaluation_data = split_data(all_data)
    vectorizer = CountVectorizer(binary=True)  
    classifier = training_step(training_data, vectorizer)

    # Text input for review
    review = st.text_input("Enter The Review", "Write Here...")

    # Button to predict sentiment for single review
    if st.button('Predict Sentiment'):
        result = print_result(analyse_text(classifier, vectorizer, review))
        st.success(result[1])
    else:
        st.write("Press the above button..")

    # Upload CSV file for bulk prediction
    uploaded_file = st.file_uploader("Upload CSV file for bulk prediction", type=['csv'])

    # Perform bulk prediction
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        corpus = data["Sentence"]
        X_prediction = vectorizer.transform(corpus).toarray()
        y_predictions = classifier.predict(X_prediction)
        data["Predicted Sentiment"] = ["Positive" if pred == '1' else "Negative" for pred in y_predictions]

        # Display predicted sentiment
        st.subheader("Predicted Sentiment for Each Sentence:")
        st.write(data)

        # Generate sentiment distribution graph
        sentiment_counts = data["Predicted Sentiment"].value_counts()
        plt.figure(figsize=(8, 6))
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
        plt.title("Sentiment Distribution")
        plt.xlabel("Sentiment")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        st.pyplot(plt)

elif nav_selection == "Home":
    # Apply CSS for color
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: #F5F5DC;
            color: black;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    st.write("Amazon Product Review Analyzer is your one-stop solution for analyzing sentiment in Amazon product reviews. We are dedicated to providing valuable insights derived from user reviews, empowering businesses and individuals to make informed decisions.")
    st.write("Our platform offers a comprehensive solution for businesses and individuals looking to make sense of the vast amount of feedback available on Amazon. Whether you're a seller looking to improve your products or a consumer researching your next purchase, our sentiment analysis tools are designed to meet your needs.")
    st.write("Thank you for choosing Amazon Product Review Analyzer!")

    st.write('\n\n')
    #st.write("Ready to analyze some reviews?")
  #  if st.button('Go to Analyzer'):
   #     nav_selection="Analyzer"