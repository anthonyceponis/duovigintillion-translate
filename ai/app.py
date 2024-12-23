from flask import Flask, request, jsonify
from flask_cors import CORS
from translate import translate_sentence

app = Flask(__name__)
CORS(app)

# Placeholder translation function (replace this with your transformer model)
def translate_text(input_text):
    # Example: simply reverses the text for demonstration
	ans = translate_sentence(input_text)
	return ans

@app.route('/translate', methods=['GET'])
def translate():
    input_text = request.args.get('text')
    if not input_text:
        return jsonify({'error': 'No input text provided'}), 400
    
    translated_text = translate_text(input_text)
    return jsonify({'translated_text': translated_text})

if __name__ == '__main__':
    app.run(debug=True)
