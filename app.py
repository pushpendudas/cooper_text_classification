# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 13:59:21 2021

@author: Pushpendu
"""

import ast
from flask import Flask, request, jsonify
from text_label_prediction import statement_prediction_main_pipeline

import warnings

warnings.filterwarnings("ignore")
app = Flask(__name__)


@app.route('/statement_prediction', methods=['POST'])
def question_identification_api():
    output = dict()
    prediction_result = []
    
    try:
        if request.method == 'POST':
            req_json = ast.literal_eval(request.get_data(as_text=True))
            statement = str(req_json["statement"])
            output['statement'] = statement

            prediction_result = statement_prediction_main_pipeline(statement)
            print("prediction_result:", prediction_result)
            
            output['predicted_label'] = prediction_result[0]
            output['confidence_score'] = prediction_result[1]

    except Exception as exp:
        print("Exceotion in statement prediction api", exp)

    return jsonify(output)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True, use_reloader=False, threaded=True)