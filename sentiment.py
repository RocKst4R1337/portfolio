import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS
import nltk
import re
import string
from nltk.corpus import stopwords
import plotly.express as px


def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text


def calculateNumOfPositive(data):
    i = 0
    for index, x in data.iterrows():
        if (x['Positive'] > x['Negative']) and (x['Positive'] > x['Neutral']):
            i = i + 1
    return i


def calculateNumOfNegative(data):
    i = 0
    for index, x in data.iterrows():
        if (x['Negative'] > x['Positive']) and (x['Negative'] > x['Neutral']):
            i = i + 1
    return i


def calculateNumOfNeutral(data):
    i = 0
    for index, x in data.iterrows():
        if (x['Neutral'] > x['Negative']) and (x['Neutral'] > x['Positive']):
            i = i + 1
    return i


# dowload data
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/flipkart_reviews.csv")

# stop words, stemmer
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
stopword=set(stopwords.words('english'))

# cleaning of reviwes
data["Review"] = data["Review"].apply(clean)

# calculating data for pie chart
ratings = data["Rating"].value_counts()
numbers = ratings.index
quantity = ratings.values

# plotting pie chart
figure = px.pie(data, values=quantity, names=numbers,hole = 0.5)
figure.show()

# generating wordcloud and plotting it
text = " ".join(i for i in data.Review)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords,background_color="white").generate(text)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# perform sentiment analysis of reviews
nltk.download('vader_lexicon')
sentiments = SentimentIntensityAnalyzer()
data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["Review"]]
data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["Review"]]
data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["Review"]]
data["Compound"] = [sentiments.polarity_scores(i)["compound"] for i in data["Review"]]
data = data[["Review", "Positive", "Negative", "Neutral", "Compound"]]

numberOfPositive = calculateNumOfPositive(data)
numberOfNegative = calculateNumOfNegative(data)
numberOfNeutral = calculateNumOfNeutral(data)

# printing number of each type of review
print("Positive: " + str(numberOfPositive) + "\n" + "Neutral: " + str(numberOfNeutral) + "\n"
      + "Negative: " + str(numberOfNegative) + "\n")

# plotting pie chart of reviews
figure = px.pie(data, values=[numberOfPositive, numberOfNeutral, numberOfNegative], names=['Positive', 'Neutral', 'Negative'],hole = 0.5)
figure.show()



