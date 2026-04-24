# Name: Jeevanraaj a/l Thayanithi
# ID: SW01083389

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel

nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv('news_dataset.csv')
df = df[['text']].dropna()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    words = text.lower().split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return words

df['tokens'] = df['text'].apply(clean_text)

dictionary = corpora.Dictionary(df['tokens'])
corpus = [dictionary.doc2bow(text) for text in df['tokens']]

lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=4, random_state=42)

for i, topic in lda.print_topics():
    print(f"Topic {i}:", topic)

coherence_model = CoherenceModel(model=lda, texts=df['tokens'], dictionary=dictionary, coherence='c_v')
coherence_score = coherence_model.get_coherence()

print("Coherence Score:", coherence_score)
