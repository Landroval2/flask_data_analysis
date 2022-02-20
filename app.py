from flask import Flask, jsonify, request, render_template
from flask import session
from flair.models import TextClassifier
from flair.data import Sentence

from models.model_factory import ModelFactory



# creating instance of the class
app = Flask(__name__, template_folder='templates')
#app.secret_key = "super_secret_key"

# # app.config['SECRET_KEY'] = 'oh_so_secret'

# app.config['MONGO_DBNAME'] = 'exposeModel'
# app.config['MONGO_URI'] = 'mongodb://localhost:27017/exposeModel'
# mongo = PyMongo(app)

classifier = TextClassifier.load('sentiment')

# to tell flask what url should trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

# prediction function
def text_classifier(input_text):

    sentence = Sentence(input_text)

    # # run classifier over sentence
    classifier.predict(sentence)

    #extract text and its prediction
    text = sentence.to_plain_string()
    label = sentence.labels[0]
    result = {'phrase': text, 'tag': label.value, 'score': round(label.score, 2)}

    return result


@app.route('/result', methods = ['POST'])
def result():

    # input_text = request.form.values()
    # input_text = list(map(str, input_text))[0]
    input_text = request.json['input_text']
    model_type = request.json['model_type']

    model = ModelFactory.get_model(model_type)
    result = model.make_prediction(input_text)
    
    #return flask.render_template("result.html", result=result)
    return result


if __name__ == '__main__':
    app.run(debug=True)