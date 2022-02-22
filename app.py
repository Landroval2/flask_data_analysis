from flask import Flask, request, render_template

from models.model_factory import ModelFactory


# creating instance of the class
app = Flask(__name__, template_folder='templates')
#app.secret_key = "super_secret_key"

# # app.config['SECRET_KEY'] = 'oh_so_secret'

# app.config['MONGO_DBNAME'] = 'exposeModel'
# app.config['MONGO_URI'] = 'mongodb://localhost:27017/exposeModel'
# mongo = PyMongo(app)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    input_text = request.json['input_text']
    model_type = request.json['model_type']

    model = ModelFactory.get_model(model_type)
    prediction = model.make_prediction(input_text)

    return prediction


if __name__ == '__main__':
    app.run(debug=True)
