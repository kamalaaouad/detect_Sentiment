import json

from flask import Flask, request
from transformers import pipeline

path = './distilbert-base-uncased-finetuned-sst-2-english'

sentiment_analysis = pipeline("sentiment-analysis", model=path)
app = Flask(__name__)

@app.route('/home', methods=['GET'])
def home():
    print('hi')
    #return jsonify(message=str("index.html"))
    language = "Python"
    company = "Oreivaton"
    Itemid = 1
    price = 0.00

    # Create Dictionary
    value = {
        "language": language,
        "company": company,
        "Itemid": Itemid,
        "price": price
    }

    # Dictionary to JSON Object using dumps() method
    # Return JSON Object
    return json.dumps(value)
    #return json.dump({'message': "index.html"})

@app.route("/addme", methods=['POST'])
def hello():
    print('hello')
    print('hello')
    data_text = request.form.get('data_text')
    res = sentiment_analysis(data_text)
    return json.dumps(res[0])
    #return json.dumps({"meg": data_text})


if __name__ == "__main__":
    from waitress import serve

    serve(app, host="0.0.0.0", port=8080)
    # app.run()
