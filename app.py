from flask import Flask, request, render_template, send_file
from icrawler.builtin import GoogleImageCrawler
from PIL import Image
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions

app = Flask(__name__)
model = MobileNet(weights='imagenet')

DOWNLOAD_DIR = 'downloads'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_images(query, max_num=20):
    crawler = GoogleImageCrawler(storage={'root_dir': DOWNLOAD_DIR})
    crawler.crawl(keyword=query, max_num=max_num, min_size=(200,200), overwrite=True)

def filter_relevant_images(paths, query):
    kept = []
    for p in paths:
        img = Image.open(p).resize((224,224))
        arr = preprocess_input(np.expand_dims(np.array(img),0))
        preds = decode_predictions(model.predict(arr), top=3)[0]
        labels = [t[1].lower() for t in preds]
        if any(query.lower() in lbl for lbl in labels):
            kept.append(p)
    return kept

def extract_color_mask(arr, color):
    # color = 'brown','black','white','blue'
    r,g,b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    if color=='white':
        mask = (r>200)&(g>200)&(b>200)
    elif color=='black':
        mask = (r<50)&(g<50)&(b<50)
    elif color=='blue':
        mask = (b>150)&(r<100)&(g<100)
    elif color=='brown':
        mask = (r>100)&(g>50)&(b<50)
    return mask

def find_common_pattern(image_paths, color, threshold_ratio=0.6):
    counts = None
    for p in image_paths:
        img = Image.open(p).convert('RGB').resize((200,200))
        arr = np.array(img)
        mask = extract_color_mask(arr, color).astype(int)
        counts = mask if counts is None else counts + mask
    limit = len(image_paths) * threshold_ratio
    common = counts >= limit
    return common  # boolean 2D array

def render_pattern(mask, output_path):
    h,w = mask.shape
    img = Image.new('RGBA', (w,h), (0,0,0,0))
    pix = img.load()
    for y in range(h):
        for x in range(w):
            if mask[y,x]:
                pix[x,y] = (255,255,255,255)  # disegno in bianco
    img.save(output_path)

@app.route('/', methods=['GET','POST'])
def index():
    if request.method=='POST':
        q = request.form['query']
        # 1. scarica
        download_images(q, max_num=20)
        files = [os.path.join(DOWNLOAD_DIR,f) for f in os.listdir(DOWNLOAD_DIR)]
        # 2. filtra
        relevant = filter_relevant_images(files, q)
        # 3. trova pattern per ogni colore
        for col in ['brown','black','white','blue']:
            mask = find_common_pattern(relevant, col)
            if mask.sum() > 100:  # se ci sono abbastanza pixel
                out = os.path.join('static','pattern.png')
                render_pattern(mask, out)
                return render_template('result.html', color=col)
        # fallback
        return render_template('result.html', color=None)
    return render_template('index.html')

@app.route('/pattern.png')
def pattern_png():
    return send_file('static/pattern.png', mimetype='image/png')

if __name__=='__main__':
    app.run(debug=True)
