# App Reviews Sentiment Analysis using Python



![image](https://github.com/user-attachments/assets/61e26f0d-109d-4eaf-8683-79b23defef49)




**Introduction**:

App Reviews Sentiment Analysis means evaluating and understanding the sentiments expressed in user reviews of mobile applications (apps). It involves using data analysis techniques to determine whether the sentiments in these reviews are positive, negative, or neutral and App Reviews Sentiment Analysis is a valuable tool for app developers and businesses to understand user feedback, prioritize feature updates, and maintain a positive user community.Throughout this project I delved into the realm of data using Pythons libraries and functions. From analyzing to visualizing the data this project covers all aspects of data processing. The interactive environment provided by Jupyter notebook enhanced my experience by allowing me to engage with the data and discover patterns.

**Aim and Objective**:

App Reviews Sentiment Analysis serves as a crucial resource for app developers and businesses to gain insights into user feedback. By analyzing sentiments expressed in reviews, developers can identify user preferences, pinpoint areas for improvement, prioritize feature enhancements, and foster a positive community among users. This project aims to equip stakeholders with actionable insights that can enhance user satisfaction and drive app success.

**Data Sources and Collection**:

The provided dataset comprises user reviews of the LinkedIn app, a professional networking platform. It includes the following columns:
Review: This column contains the actual text of the user review. It offers qualitative insights into the user’s experience with the app, highlighting praises, complaints, suggestions, and general comments.
Rating: Numerical ratings ranging from 1 to 5 accompany the reviews, with 1 being the lowest and 5 being the highest. These ratings provide a quantitative measure of user satisfaction.

LinkedIn Dataset : [Dataset](https://statso.io/sentiment-analysis-case-study/)

**Exploratory Data Analysis (EDA)**:

Performing Exploratory Data Analysis (EDA) on app reviews is a crucial step in understanding user sentiment and gathering insights that can inform decisions about app improvements. 
Here’s a detailed breakdown of how you can analyze the length of reviews, ratings, and other relevant factors in EDA:

**Review Lengths**: Discuss the average review length and how it varies with ratings.

**Ratings Distribution**: Highlight how ratings are distributed across different review lengths.

**Common Themes**: Identify frequently mentioned words that could indicate common user experiences or issues.

**Tools and Libraries Used**🛠️:

**Programming Language**: Python

**Libraries**: Pandas, Matplotlib, Seaborn, Textblob and Wordcloud

**IDE**: Jupyter Notebook

**Key Findings and Insights**:

Based on the analysis of the word clouds and bar chart, here are the key findings and insights:

Sentiment Analysis:
Positive sentiment dominates: Across all ratings, positive sentiment is the most prevalent, indicating that users generally had positive experiences with the product or service.

Negative sentiment is concentrated in lower ratings: Most negative sentiment is associated with ratings 1 and 2, suggesting that users are more likely to express negative feelings when they give lower ratings.

Neutral sentiment is more evenly distributed: Neutral sentiment is found across all ratings, indicating a mix of positive and negative aspects in user experiences.

Key Themes and Aspects:
Performance issues: "Slow," "load," "force close," and "crash" were frequently mentioned in negative reviews, highlighting performance problems as a major concern.

Updates and new features: Both positive and negative reviews mentioned updates and new features, suggesting that these updates had a mixed impact on user experiences.

Functionality: "Work" and "app" were mentioned across all sentiment categories, indicating that the app's functionality was a key factor in user satisfaction.

Internet connectivity: "Internet" and "connection" were mentioned in negative reviews, suggesting that network issues contributed to negative experiences.

Device compatibility: "Galaxy" and "HTC" were mentioned in multiple reviews, indicating that specific devices or models might have experienced more problems.

User Needs and Expectations:

Updates and improvements: Users expressed a need for updates and improvements to the app or product, suggesting that there is room for further development.

Better performance: The frequent mention of performance issues indicates that users value a smooth and responsive experience.

Improved functionality: The focus on the app's functionality suggests that users expect the app to work effectively and meet their needs.

**Import Required Libraries**:
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

**Exploring the Datasets**:
```python
# load the dataset
linkedin_data = pd.read_csv(r'C:\Users\neela\Desktop\linkedin-reviews.csv')

# display the first few rows of the dataset
linkedin_data.head()
```
![image](https://github.com/user-attachments/assets/6b467a60-5bf0-4143-9104-c7631342c29f)

```python
# provided the summary of the dataframe named linkedin_data
linkedin_data.info()
```
![image](https://github.com/user-attachments/assets/39f69e52-f075-4add-9bf3-849820a61b46)

**Exploratory Data Analysis**:
Now, let’s explore this data step by step. I started by analyzing the distribution of ratings. It provided insight into the overall sentiment of the reviews. 
```python
#plotting the distribution of rating

# Set the visual style for the plot using Seaborn's whitegrid style
sns.set(style='whitegrid')

# Create a figure for the plot and set the size to 9 inches by 5 inches
plt.figure(figsize=(9,5))

# Generate a count plot for the 'Rating' column in the 'linkedin_data' DataFrame
# This will show the count of each unique value in the 'Rating' column as bars
sns.countplot(data = linkedin_data ,x ='Rating')


# Add a title to the plot
plt.title('Distribution of Ratings')

# Label the x-axis as 'Rating' to indicate what is being measured on this axis
plt.xlabel('Rating')

# Label the y-axis as 'Count' to indicate the number of occurrences of each rating
plt.ylabel('Count')

# Display the plot on the screen
plt.show()
```
![image](https://github.com/user-attachments/assets/4307f2c0-5d70-4f61-8a6f-cc3a3c197e49)
Next, I analyzed the length of the reviews, as this can sometimes correlate with the sentiment or detail of feedback. I calculated the length of each review and then visualize the data:
```python
# Calculating the length of each review
linkedin_data['Review Length'] = linkedin_data['Review'].apply(len)

# Plotting the distribution of review lengths
plt.figure(figsize=(9, 6))
# Creating the Histogram and KDE Plot
sns.histplot(linkedin_data['Review Length'], bins=50,kde=True)

# Adding Title and Labels
plt.title('Distribution of Review Lengths')
plt.xlabel('Length of Review')
plt.ylabel('Count')

# Display the plot on the screen
plt.show()
```
![image](https://github.com/user-attachments/assets/945ce6c7-8b55-4fe1-a6c2-e7744a4a9102)

**Adding Sentiment Labels in the Data**:
Utilized Textblob for this task. TextBlob provides a polarity score ranging from -1 (very negative) to 1 (very positive) for a given text. I used this score to classify each review’s sentiment as positive, neutral, or negative.
```python
from textblob import TextBlob

def textblob_sentiment_analysis(review):
    # Analyzing the sentiment of the review
    sentiment = TextBlob(review).sentiment
    # Classifying based on polarity
    if sentiment.polarity > 0.1:
        return 'Positive'
    elif sentiment.polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

# Applying TextBlob sentiment analysis to the reviews
linkedin_data['Sentiment'] = linkedin_data['Review'].apply(textblob_sentiment_analysis)

# Displaying the first few rows with the sentiment
print(linkedin_data.head())
```
![image](https://github.com/user-attachments/assets/e74689c0-0eb8-4c7b-92cb-f16ac058777e)

**Analyzing App Reviews Sentiments**:
Now that our dataset is labelled, performed app reviews sentiment analysis. I began by analyzing the distribution of sentiments across the dataset. It gave us a basic understanding of the general sentiment tendency in the reviews:
```python
# Analyzing the distribution of sentiments
sentiment_distribution = linkedin_data['Sentiment'].value_counts()

# Plotting the distribution of sentiments
plt.figure(figsize=(9, 5))
sns.barplot(x=sentiment_distribution.index, y=sentiment_distribution.values)
plt.title('Distribution of Sentiments')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
```
![image](https://github.com/user-attachments/assets/c8d1fa5c-ffbd-48df-9812-95ad282512b3)

So, although the app has low ratings, still the reviewers don’t use many negative words in the reviews for the app.

Next,explored the relationship between the sentiments and the ratings. This analysis helped us understand whether there is a correlation between the sentiment of the text and the numerical rating. 
```python
sns.set(style='darkgrid')
plt.figure(figsize=(10, 5))
sns.countplot(data=linkedin_data, x='Rating', hue='Sentiment')
plt.title('Sentiment Distribution Across Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.legend(title='Sentiment')
plt.show()
```
![image](https://github.com/user-attachments/assets/f14ad388-659a-4a6c-876c-810fae75f816)

Performed a text analysis to identify common words or themes within each sentiment category. It involved examining the most frequently occurring words in positive, negative, and neutral reviews using a word cloud:
```python
pip install wordcloud
from wordcloud import WordCloud
# Function to generate a word cloud for a given sentiment
def generate_word_cloud(sentiment):
    text = ' '.join(linkedin_data[linkedin_data['Sentiment'] == sentiment]['Review'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(9,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'{sentiment} Reviews')
    plt.axis('off')
    plt.show()

# Generate word clouds for each sentiment
for sentiment in ['Positive', 'Negative', 'Neutral']:
    generate_word_cloud(sentiment)
```
![image](https://github.com/user-attachments/assets/ac44b7a6-1a15-4e18-a44b-31752a1d94ed)

![image](https://github.com/user-attachments/assets/5ad9186b-f438-439a-826e-3b0ee8a25127)

![image](https://github.com/user-attachments/assets/b9271ed3-5281-4c5a-8daf-52e9fdbebf79)











