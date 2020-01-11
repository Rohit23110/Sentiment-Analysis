import nltk
from nltk.corpus import stopwords
import pandas as pd
import random
import re
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import FreqDist, classify, NaiveBayesClassifier
import pickle

# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

def remove_stop_words(corpus):
    english_stop_words = stopwords.words('english')
    removed_stop_words = []
    for review in corpus:
        removed_stop_words.append(
            ' '.join([word for word in review.split()
                      if word not in english_stop_words])
        )
    return removed_stop_words


def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence


def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token


def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


def train_model():
    tweets = pd.read_csv(r'C:\Users\Rohit Pai\Downloads\twitter-airline-sentiment\Tweets.csv')
    tweets = tweets[['text', 'airline_sentiment']]

    tweets_list = list(tweets['text'])
    i = 0
    for tweet in tweets_list:
        tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+://\S+)", " ", tweet).split())
        tweets_list[i] = tweet.lower()
        i = i + 1
    tweets_list = remove_stop_words(tweets_list)

    # TODO: number hatao try

    tokenizer = nltk.TweetTokenizer()
    new_tweets_list = []
    for tweet in tweets_list:
        token = tokenizer.tokenize(tweet)
        new_tweets_list.append(lemmatize_sentence(token))
    print(new_tweets_list)

    positive_tweet_tokens = []
    negative_tweet_tokens = []
    neutral_tweet_tokens = []

    for i in range(0, len(tweets)):
        if tweets.iloc[i]['airline_sentiment'] == 'positive':
            positive_tweet_tokens.append(new_tweets_list[i])
        elif tweets.iloc[i]['airline_sentiment'] == 'negative':
            negative_tweet_tokens.append(new_tweets_list[i])
        else:
            neutral_tweet_tokens.append(new_tweets_list[i])

    all_pos_words = get_all_words(positive_tweet_tokens)
    all_neg_words = get_all_words(negative_tweet_tokens)
    all_neu_words = get_all_words(neutral_tweet_tokens)

    freq_dist_pos = FreqDist(all_pos_words)
    print(freq_dist_pos.most_common(10))
    freq_dist_neg = FreqDist(all_neg_words)
    print(freq_dist_neg.most_common(10))
    freq_dist_neu = FreqDist(all_neu_words)
    print(freq_dist_neu.most_common(10))

    positive_tokens_for_model = get_tweets_for_model(positive_tweet_tokens)
    negative_tokens_for_model = get_tweets_for_model(negative_tweet_tokens)
    neutral_tokens_for_model = get_tweets_for_model(neutral_tweet_tokens)

    positive_dataset = [(tweet_dict, "positive")
                        for tweet_dict in positive_tokens_for_model]

    negative_dataset = [(tweet_dict, "negative")
                        for tweet_dict in negative_tokens_for_model]

    neutral_dataset = [(tweet_dict, "neutral")
                       for tweet_dict in neutral_tokens_for_model]

    dataset = positive_dataset + negative_dataset + neutral_dataset

    random.shuffle(dataset)

    train_data = dataset[:10000]
    test_data = dataset[10000:]

    classifier = NaiveBayesClassifier.train(train_data)

    print("Accuracy is:", classify.accuracy(classifier, test_data))

    print(classifier.show_most_informative_features(10))

    pickle.dump(classifier, open('sentiment_model.sav', 'wb'))


if __name__ == "__main__":
    train_model()
