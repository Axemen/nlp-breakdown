from flask import Flask, render_template, request, redirect, Markup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
import en_core_web_sm
from nltk.corpus import stopwords
from spacy import displacy

import pandas as pd
import plotly
import plotly.express as px
from collections import Counter

nlp = en_core_web_sm.load()
stop_words = set(stopwords.words('english'))
stop_words.add(('-pron-', "'s", '\r\n\r\n', '\r\n'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/results',  methods = ['GET', 'POST'])
def landing():
    processed_dict = {}

    if request.method == "POST":

        user_input = request.form['input_text']

        doc = nlp(user_input)

        processed_dict['sentences'] = [i.text for i in doc.sents]
        processed_dict['tokens'] = [[j.text for j in i if j.pos_ is not 'PUNCT'] for i in doc.sents]
        processed_dict['pos_lists'] = [[(i.text, spacy.explain(i.pos_)) for i in j] for j in doc.sents]
        processed_dict['lemma_list'] = [[i.lemma_ for i in j if i.pos_ is not 'PUNCT'] for j in doc.sents]
        processed_dict['filtered_stop_words'] = [[i.lemma_ for i in j if (i.lemma_.lower() not in stop_words) and (i.pos_ is not 'PUNCT')] for j in doc.sents]
        processed_dict['noun_chunks'] = [[i.text, i.root.text, spacy.explain(i.root.dep_), i.root.head.text] for i in doc.noun_chunks]
        processed_dict['named_entities'] = [[ent.text, ent.label_, spacy.explain(ent.label_)] for ent in doc.ents]

        ent_html = Markup(displacy.render(doc, style = 'ent'))

        token_cnt = Counter()
        for token in doc:
            if (token.text.lower() not in stop_words) and (token.pos_ is not 'PUNCT'):
                token_cnt[token.text] += 1

        df = pd.DataFrame(token_cnt.most_common(10)).rename(columns = {0:'word', 1:'count'})
        df.head()
        plot = px.bar(df, x = 'word', y = 'count')

        token_count_bar = Markup(plotly.offline.plot(plot, output_type = 'div'))

        return render_template('index.html', 
            user_input = user_input, 
            processed_dict = processed_dict, 
            ent_html = ent_html,
            token_count_bar = token_count_bar)

    return render_template('landing.html')

if __name__ == "__main__":
    app.run(debug = True)


