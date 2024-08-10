from flask import Flask
import monkeyModel
import violenceModel

app = Flask(__name__)

@app.route('/')
def index():
    return 'Hello World'

@app.route('/<name>')
def print_name(name):
    return 'Hi there, {}'.format(name)

@app.route('/test/<filePath>')
def test(filePath):
    print('file path ' + filePath)
    return monkeyModel.mainMethod(filePath)

@app.route('/violence/<filePath>')
def violence(filePath):
    print('file path ' + filePath)
    return violenceModel.mainMethod(filePath)

if __name__ == '__main__':
    app.run(debug=True)