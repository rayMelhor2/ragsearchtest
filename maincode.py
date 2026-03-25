from projLogics import *
from flask import Flask, render_template, request, jsonify
import os

chromadbpath = './chromadb' #путь хранения базы данных всех книг
rerankerModel = './models/Qwen3-Reranker-0.6B' #реранкер модель
embeddingsModel = './models/paraphrase-multilingual-MiniLM-L12-v2' #векторная модель
llmModel = './models/Qwen3-1.7B'

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        if request.files['file'].filename[-4:] == '.txt':
            file = request.files['file']
            print(file.filename.split('.txt')[0])
            file.save(os.path.join('books', file.filename))
            chromadbadd(f'./books/{file.filename}')
            result = f'Успешно добавил книгу с названием {file.filename}'
        else:
            result = 'Этот файл не .txt'

    papkoFiles = os.listdir('./books')
    files = []
    for file in papkoFiles:
        files.append(file.split('.')[0])
    return render_template('index.html', result=result, files=files)


@app.route('/search', methods=['POST'])
def search():
    query = request.json['query']
    bibilbab = reranksearch(query)
    llmthink = LLMsearch(SearchQuery=query, DOCSSWAT=bibilbab)
    result = f'{llmthink}\n\n\n'
    for otvet in bibilbab:
        file = otvet['metadata']['filename']
        file = file.strip('/')
        result += f"Цитата: {otvet['document']}\n"
        result += f"Файл: {file}, Абзац: {otvet['metadata']['abzac']}\n"
        result += f"Точность ответа: {round(otvet['score']*100, 4)}%\n\n\n"
    print(result)
    return jsonify({ 'result': result })

if __name__ == '__main__':
    app.run(debug=True)
