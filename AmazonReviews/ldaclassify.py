import json
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from multiprocessing import cpu_count

# this should be your checkpoint #2 file
REVIEWS_FILE = 
# output files
CLASSIFIED_REVIEWS_FILE = 'classified_reviews.jsonl'
TOPICS_FILE = 'lda_topics.txt'

N_TOPICS = 25
CORES = cpu_count()
TERMS_PER_TOPIC_LABEL = 5
STOP_WORDS = stopwords.words('english')
BUILD_VIZ = False # note: pyLDAvis is heavy - don't run it unless you need a new one
if BUILD_VIZ:
    import pyLDAvis
    import pyLDAvis.sklearn


review_count = 0
def load_reviews():
    global review_count
    texts = set()
    reviews = json.load(open(REVIEWS_FILE, 'r'))
    for asin, data in reviews.items():
        if 'reviewText' in data:
            reviewtext = data['reviewText']
            summary = data['summary']
            asin = data['asin']
            review = '%s %s %s' % (asin, summary, reviewtext)
            if review not in texts:
                texts.add(review)
                review_count += 1
                yield review


print('Building tf-idf data ..')
vectorizer = TfidfVectorizer(stop_words=STOP_WORDS)
X = vectorizer.fit_transform(load_reviews())
print('.. done')
print('Building %d topics ..' % N_TOPICS)
lda_tfidf = LatentDirichletAllocation(n_components=N_TOPICS, n_jobs=CORES)
lda_tfidf.fit(X)
print('.. done')

if BUILD_VIZ:
    print('Building LDA visualization ..')
    p = pyLDAvis.sklearn.prepare(lda_tfidf, X, vectorizer)
    pyLDAvis.save_html(p, 'pyLDAvis.html')
    print('.. done writing pyLDAvis.html')

topics = []
def write_topics_file():
    print('Writing topics to: %s' % TOPICS_FILE)
    with open(TOPICS_FILE, 'w') as topics_file:
        lexicon = vectorizer.get_feature_names()
        n_top_words = TERMS_PER_TOPIC_LABEL
        for topic_idx, topic in enumerate(lda_tfidf.components_):
            topic = topic.argsort()[:-n_top_words -1:-1]
            feature_names = []
            for i in topic:
                feature_names.append(lexicon[i])
            topic_string = " ".join(feature_names)
            topics_file.write(topic_string + '\n')
            print('Topic %d: %s' % (topic_idx, topic_string))
            topics.append(topic_string)
    print('.. done writing topics file')
write_topics_file()


def write_classified_docs():
    print('Writing classified reviews to: %s' % CLASSIFIED_REVIEWS_FILE)
    with open(CLASSIFIED_REVIEWS_FILE, 'w') as outfile:
        for i, review in enumerate(load_reviews()):
            if not i % 1000:
                print('%d of %d reviews' % (i, review_count))
            Y = vectorizer.transform([review])
            prediction = lda_tfidf.transform(Y)
            predictions_list = prediction.tolist()[0]
            high_score = max(predictions_list)
            topic_id = predictions_list.index(high_score)
            outfile.write('%s\n' % json.dumps({
                'review': review,
                'topic': {
                    'id': topic_id,
                    'label': topics[topic_id]
                }
            }))
    print('.. done')
write_classified_docs()
