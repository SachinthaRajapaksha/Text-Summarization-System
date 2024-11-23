import PyPDF2
import nltk
import networkx as nx
import shap
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import Ridge


nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


# Read PDF data and summarize
def process_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


#Loading fine-tuned summarization model for abstractive summarization
summarizer = pipeline('summarization', model='finetuned_summarizer')


#implementing pagerank algorithm for extractive summarization
def pagerank_summarize_with_shap(text, num_sentences=5):
    sentences = sent_tokenize(text)
    vectorizer = TfidfVectorizer()
    sentence_vectors = vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(sentence_vectors)
    graph = nx.from_numpy_array(similarity_matrix)
    
    # PageRank algorithm to score sentences
    scores = nx.pagerank(graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary_sentences = [s for _, s in ranked_sentences[:num_sentences]]
    summary = " ".join(summary_sentences)
    
    #implement linear model for SHAP
    model = Ridge(alpha=1.0)
    model.fit(sentence_vectors.toarray(), list(scores.values()))
    
    explainer = shap.LinearExplainer(model, sentence_vectors.toarray(), feature_perturbation="correlation_dependent")
    shap_values = explainer.shap_values(sentence_vectors.toarray())
    
    shap_sentence_values = shap_values.mean(axis=1).tolist()
    
    #selecting top contributing sentences
    shap_sentence_pairs = sorted(zip(sentences, shap_sentence_values), key=lambda x: x[1], reverse=True)
    
    # Return top 3 contributing sentences
    top_contributing_sentences = shap_sentence_pairs[:3]
    top_sentences, top_shap_values = zip(*top_contributing_sentences)
    
    return summary, list(top_shap_values), list(top_sentences)

# Modify summarize_text to include SHAP
def summarize_text(text, summary_type, method='combined'):
    length = len(text.split())

    if summary_type == 'short':
        num_sentences = 3
        max_length = int(length / 8)
        min_length = int(length / 10)
    elif summary_type == 'medium':
        num_sentences = 5
        max_length = int(length / 4)
        min_length = int(length / 5)
    elif summary_type == 'detailed':
        num_sentences = 7
        max_length = int(length / 2)
        min_length = int(length / 3)
    else:
        raise ValueError("Invalid summary type. Choose from 'short', 'medium', or 'detailed'.")

    if method == 'extractive':
        summary, shap_values, sentences = pagerank_summarize_with_shap(text, num_sentences=num_sentences)
        return summary, shap_values, sentences
    elif method == 'abstractive':
        summary = summarizer(text, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4)
        return summary[0]['summary_text'], None, None
    elif method == 'combined':
        extractive_summary, shap_values, sentences = pagerank_summarize_with_shap(text, num_sentences=num_sentences)
        combined_text = f"{extractive_summary}\n\nAbstractive summary:"
        abstractive_summary = summarizer(combined_text,max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4,early_stopping=True)
        return f"Extractive summary:\n{extractive_summary}\n\nAbstractive summary:\n{abstractive_summary[0]['summary_text']}", shap_values, sentences
    else:
        raise ValueError("Invalid method. Choose from 'abstractive', 'extractive', or 'combined'.")

#implementing keyword extraction
def extract_keywords(text, ngram_range=(1, 3), top_n=20):
    # Tokenize the text and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    
    # Join tokens into a single string for TfidfVectorizer
    text = ' '.join(tokens)

    # Create a TfidfVectorizer to generate n-grams
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=top_n)

    # Fit and transform the text using TF-IDF
    tfidf_matrix = vectorizer.fit_transform([text])
    
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1
    tfidf_scores = dict(zip(feature_names, scores))

    sorted_keywords = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)

    top_keywords = [keyword for keyword, score in sorted_keywords[:top_n]]

    return top_keywords


# topic modelling using our model
model_name_topics = "topic_model_t5"
topic_model = pipeline("text2text-generation", model=model_name_topics)

def generate_topics(text):
    generated_topics = topic_model(text, max_length=50, num_return_sequences=5, do_sample=True, top_k=50, top_p=0.95)
    topics = [topic['generated_text'] for topic in generated_topics]
    return topics


# Sentiment analysis using VADER
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    sentiment_scores = analyzer.polarity_scores(text)
    compound_score = sentiment_scores['compound']
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

def sentiment_summary(article):
    sentiment = analyze_sentiment(article)
    if sentiment == 'positive':
        summary = "Document you entered reflects an overall 'Positive' narrative about the highlighted subject."
    elif sentiment == 'negative':
        summary = "Document you entered reflects an overall 'Negative' narrative about the highlighted subject."
    else:
        summary = "Document you entered reflects an overall 'Neutral' narrative about the highlighted subject."
    return summary