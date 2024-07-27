from flask import Flask
import monkeyModel

app = Flask(__name__)

@app.route('/')
def index():
    return 'Hello World'

@app.route('/<name>')
def print_name(name):
    return 'Hi there, {}'.format(name)

@app.route('/test/<filePath>')
def test(filePath):
    return monkeyModel.mainMethod(filePath)

if __name__ == '__main__':
    app.run(debug=True)