from flask import Flask, render_template, request, redirect, url_for
import os
import re
from transformers import pipeline

app = Flask(__name__)

# Đường dẫn lưu các file
HATE_WORDS_FILE = 'data/hate_speech_words.txt'
UPLOAD_FOLDER = 'uploads'

# Đảm bảo thư mục 'uploads' và 'data' tồn tại
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(HATE_WORDS_FILE), exist_ok=True)

# Tạo pipeline AI để phát hiện ngôn ngữ thù hận
# Sử dụng mô hình 'unitary/toxic-bert' để phát hiện ngôn ngữ thù hận
hate_speech_detector = pipeline("text-classification", model="unitary/toxic-bert")

# Hàm đọc danh sách từ ngữ thù hận từ tệp
def load_hate_words():
    if os.path.exists(HATE_WORDS_FILE):
        with open(HATE_WORDS_FILE, 'r', encoding='utf-8') as f:
            return set([word.strip() for word in f.read().split(',') if word.strip()])
    return set()

# Hàm lưu danh sách từ ngữ thù hận vào tệp
def save_hate_words(new_words):
    hate_words = load_hate_words()
    unique_words = set(new_words) - hate_words  # Chỉ lưu từ mới không trùng
    if unique_words:
        with open(HATE_WORDS_FILE, 'a', encoding='utf-8') as f:
            if hate_words:
                f.write(',' + ','.join(unique_words))
            else:
                f.write(','.join(unique_words))
    return list(unique_words)

# Hàm phát hiện ngôn ngữ thù hận từ văn bản
def detect_hate_speech(text):
    hate_words = load_hate_words()
    # Tách văn bản thành câu theo dấu chấm, ?, :, ;, / và xuống dòng
    sentences = re.split(r'(?<=[\.\?\:\;\n])\s+', text)
    highlighted_sentences = []

    for sentence in sentences:
        original_sentence = sentence  # Lưu câu gốc
        for hate_word in hate_words:
            if hate_word in sentence:
                # Tô màu đỏ và in đậm cho từ ngữ thù hận
                sentence = sentence.replace(hate_word, f'<span style="color: red; font-weight: bold;">{hate_word}</span>')
        if sentence != original_sentence:  # Chỉ thêm những câu có chứa từ ngữ thù hận
            highlighted_sentences.append(sentence.strip())

    # Trả về các câu chứa từ ngữ thù hận, mỗi câu xuống dòng
    return '<br><br>'.join(highlighted_sentences)

@app.route('/')
def index():
    return render_template('index.html')

# Route để thêm từ ngữ thù hận
@app.route('/add_hate_words', methods=['GET', 'POST'])
def add_hate_words():
    if request.method == 'POST':
        if 'hate_words_file' not in request.files:
            return redirect(request.url)

        file = request.files['hate_words_file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            # Đọc từ file chứa từ ngữ thù hận
            with open(file_path, 'r', encoding='utf-8') as f:
                new_words = [word.strip() for word in f.read().split(',')]
                added_words = save_hate_words(new_words)

            return render_template('add_hate_words.html', message="Đã thêm từ ngữ thù hận mới.", added_words=added_words)
    return render_template('add_hate_words.html')

# Route để phân tích file văn bản
@app.route('/analyze_file', methods=['GET', 'POST'])
def analyze_file():
    result = ""
    if request.method == 'POST':
        if 'text_file' not in request.files:
            return redirect(request.url)

        file = request.files['text_file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            # Đọc file văn bản
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Sử dụng mô hình AI để phát hiện ngôn ngữ thù hận
            result = detect_hate_speech(text)

    return render_template('analyze_file.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
