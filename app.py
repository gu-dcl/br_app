
import os
import sys
import zipfile
import tempfile
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, dash_table
from dash.exceptions import PreventUpdate
import base64
import io
from gensim import corpora, models
from gensim.models.ldamodel import LdaModel
from gensim.models import TfidfModel
import pyLDAvis
from datetime import datetime.gensim_models as gensimvis
import pyLDAvis
from datetime import datetime

# Set up dynamic file paths
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
TOPIC_DATA_PATH = os.path.join(MODEL_DIR, "dominant_topic_per_doc.csv")

os.makedirs(MODEL_DIR, exist_ok=True)

app = Dash(__name__)
app.title = "LDA Topic Explorer"

def preprocess_and_model(df):
    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS as sw

    nlp = spacy.load("en_core_web_sm")
    my_sw = ["introduction", "part", "forword", "book", "quote", "telos", "journal", "essay", "chapter", "epilogue", " ", "like", "editorial"]
    for w in my_sw:
        sw.add(w)

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Decade"] = (df["Year"] // 10) * 10
    df['tokens'] = df['Title+Abstract'].apply(lambda text: ' '.join(
        [token.text for token in nlp(text) if not token.is_stop and not token.is_punct and not token.text.isdigit() and len(token.text) > 1]))
    corpus_text = df['tokens'].apply(lambda text: [token.text for token in nlp(text)]).tolist()

    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    count_vectorizer = CountVectorizer(stop_words='english', max_features=50)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=50)

    word_count_terms = []
    tfidf_terms = []

    for decade in df['Decade'].unique():
        decade_docs = df[df['Decade'] == decade]['Title+Abstract']
        X_word_freq = count_vectorizer.fit_transform(decade_docs)
        word_frequencies = X_word_freq.sum(axis=0).A1
        word_terms = count_vectorizer.get_feature_names_out()
        word_freq_df = pd.DataFrame(list(zip(word_terms, word_frequencies)), columns=['term', 'frequency'])
        top_word_terms = word_freq_df.sort_values(by='frequency', ascending=False).head(25)['term'].tolist()

        X_tfidf = tfidf_vectorizer.fit_transform(decade_docs)
        tfidf_scores = X_tfidf.sum(axis=0).A1
        tfidf_terms_list = tfidf_vectorizer.get_feature_names_out()
        tfidf_df = pd.DataFrame(list(zip(tfidf_terms_list, tfidf_scores)), columns=['term', 'tfidf'])
        top_tfidf_terms = tfidf_df.sort_values(by='tfidf', ascending=False).head(25)['term'].tolist()

        word_count_terms.append(top_word_terms)
        tfidf_terms.append(top_tfidf_terms)

    comparison_df = pd.DataFrame({
        'word count': word_count_terms,
        'tfidf': tfidf_terms
    }, index=df['Decade'].unique())
    comparison_df.to_csv(os.path.join(MODEL_DIR, "top_terms_per_decade.csv"), encoding="utf-8-sig", index_label='Decade')

    dictionary = corpora.Dictionary(corpus_text)
    dictionary.filter_extremes(no_below=2, no_above=0.5)
    corpus_bow = [dictionary.doc2bow(text) for text in corpus_text]

    tfidf_model = TfidfModel(corpus_bow)
    corpus_tfidf = tfidf_model[corpus_bow]

    top_terms_per_doc = []
    for i, doc in enumerate(corpus_tfidf):
        sorted_terms = sorted(doc, key=lambda x: x[1], reverse=True)[:5]
        top_terms = [(dictionary[id], round(weight, 3)) for id, weight in sorted_terms]
        formatted = ", ".join([f"{term} ({weight})" for term, weight in top_terms])
        top_terms_per_doc.append({"Document #": i + 1, "Top Terms": formatted})
    top_terms_df = pd.DataFrame(top_terms_per_doc)
    top_terms_df.to_csv(os.path.join(MODEL_DIR, "top_tfidf_terms_per_doc.csv"), encoding="utf-8-sig", index=False)

    lda_model = LdaModel(corpus=corpus_bow, id2word=dictionary, num_topics=5, passes=10)
    topics_data = []
    for topic_id in range(lda_model.num_topics):
        words = lda_model.show_topic(topic_id, topn=25)
        word_list = [word for word, _ in words]
        short_label = ", ".join(word_list[:5])
        full_label = ", ".join(word_list)
        topics_data.append({"Topic ID": topic_id, "Label (Top 5)": short_label, "Top 25 Words": full_label})
    topics_df = pd.DataFrame(topics_data)
    topics_df.to_csv(os.path.join(MODEL_DIR, "lda_topics_top25_and_labels.csv"), encoding="utf-8-sig", index=False)

    doc_topic_data = []
    for i, doc_bow in enumerate(corpus_bow):
        topic_probs = lda_model.get_document_topics(doc_bow, minimum_probability=0.0)
        sorted_topics = sorted(topic_probs, key=lambda x: x[1], reverse=True)
        dominant_topic_id, dominant_prob = sorted_topics[0]
        label = ", ".join([word for word, _ in lda_model.show_topic(dominant_topic_id, topn=5)])
        doc_topic_data.append({
            "Doc ID": i,
            "Dominant Topic": f"Topic {dominant_topic_id}: {label}",
            "Topic Weight": round(dominant_prob, 3),
            "Original Text": df['Title+Abstract'].iloc[i],
            "Year": df["Year"].iloc[i]
        })
    df_dominant = pd.DataFrame(doc_topic_data)
    df_dominant.to_csv(TOPIC_DATA_PATH, encoding="utf-8-sig", index=False)

    vis_data = gensimvis.prepare(lda_model, corpus_bow, dictionary)
    pyLDAvis.save_html(vis_data, os.path.join(MODEL_DIR, "lda_visualization.html"))

    return df_dominant


def create_results_zip():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"lda_results_{timestamp}.zip"
    zip_path = os.path.join(MODEL_DIR, zip_name)
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for file in os.listdir(MODEL_DIR):
            if file.endswith(".csv") or file.endswith(".html"):
                zf.write(os.path.join(MODEL_DIR, file), arcname=file)
    return zip_path


def is_first_run():
    return not os.path.exists(TOPIC_DATA_PATH)

app.layout = html.Div([
    html.H2("LDA Topic Explorer", style={"textAlign": "center"}),

    html.Div([
        dcc.Upload(id='upload-data', children=html.Button('Upload CSV', style={'margin': '10px'}), multiple=False),
        html.Button("Regenerate Topics", id="regenerate-btn", n_clicks=0, style={"margin": "10px"}),
        html.Button("Download Results", id="download-btn", n_clicks=0, style={"margin": "10px"}),
        dcc.Download(id="download-output"),
        dcc.Loading(id="loading", children=[html.Div(id="loading-output")], type="circle")
    ], style={"textAlign": "center"}),

    html.Div(id="table-container")
])

@app.callback(
    Output("table-container", "children"),
    Output("loading-output", "children"),
    Input("regenerate-btn", "n_clicks"),
    Input("upload-data", "contents"),
    State("upload-data", "filename")
)
def handle_actions(n_clicks, contents, filename):
    zip_file_path = None

    if triggered_id == "upload-data" and contents:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        df.to_csv(DATA_PATH, index=False)
        df_dominant = preprocess_and_model(df)
        zip_file_path = create_results_zip()
    elif triggered_id == "regenerate-btn":
        df = pd.read_csv(DATA_PATH)
        df_dominant = preprocess_and_model(df)
        zip_file_path = create_results_zip()
    elif is_first_run():
        df = pd.read_csv(DATA_PATH)
        df_dominant = preprocess_and_model(df)
        zip_file_path = create_results_zip()
    else:
        df_dominant = pd.read_csv(TOPIC_DATA_PATH)

    table = dash_table.DataTable(
        data=df_dominant[["Original Text", "Dominant Topic"]].to_dict("records"),
        columns=[{"name": i, "id": i} for i in ["Original Text", "Dominant Topic"]],
        page_size=10,
        style_table={"overflowX": "auto"}
    )

    if zip_file_path:
        return table, dcc.send_file(zip_file_path)
    return table, ""
@app.callback(
    Output("download-output", "data"),
    Input("download-btn", "n_clicks"),
    prevent_initial_call=True
)
def download_results(n_clicks):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
        with zipfile.ZipFile(tmp_zip.name, 'w') as zf:
            for file in os.listdir(MODEL_DIR):
                if file.endswith(".csv") or file.endswith(".html"):
                    zf.write(os.path.join(MODEL_DIR, file), arcname=file)
        return dcc.send_file(tmp_zip.name)

if __name__ == "__main__":
    app.run_server(debug=True)
