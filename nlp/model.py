import re

import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# load trained nlp, vectorizer, and cleaning function
model = joblib.load('model/job_classifier.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')


def cleaning(description):
    stop_words = set(stopwords.words('english'))
    lemma = WordNetLemmatizer()

    description = re.sub(r'[^A-z]', ' ', description).lower()
    description = word_tokenize(description)
    description = [word for word in description if word not in stop_words]
    description = [lemma.lemmatize(word) for word in description]
    description = [word for word in description if len(word) > 2]

    return ' '.join(description)


def predict_job_title(description):
    # preprocess input description
    clean_description = cleaning(description)
    vectorized_description = vectorizer.transform([clean_description])

    # make prediction
    predicted_title = model.predict(vectorized_description)

    return predicted_title[0]


def main() -> None:
    raise NotImplementedError('No Code')


if __name__ == '__main__':
    main()
