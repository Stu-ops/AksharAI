  <h1>Hindi OCR API</h1>
  <p>An easy‑to‑use OCR service for handwritten Hindi text. Built with <strong>FastAPI</strong>, <strong>OpenCV</strong>, and <strong>TensorFlow/PyTorch</strong>, it detects word regions, extracts text, and classifies snippets.
  </p>
  <hr>

  <h2>🚀 Features</h2>
  <ul>
    <li><strong>Word Detection</strong>: Highlights words in uploaded images</li>
    <li><strong>Text Extraction</strong>: Uses your custom OCR script to extract Hindi text (Devanagari)</li>
    <li><strong>Model Prediction</strong>: Classifies text snippets via your <code>.pth</code> or <code>.keras</code> model</li>
    <li><strong>Safe Temp‑File Handling</strong>: Works on Windows &amp; Linux</li>
  </ul>

  <hr>

  <h2>📂 Repository Structure</h2>
<pre><code>
hindi-ocr/
├── app.py                                  # FastAPI application  
├── requirements.txt                        # Python dependencies  
├── models/                                 # Model weight files  
│   │───notebooks/                          # Jupyter notebooks for trained model  
│   │   │── handwritten[pytorch].ipynb      # model train using pytorch
│   │   └── handwritten[tensorflow].ipynb   # model train using tensorflow
│   ├── hindi_ocr_model.pth                 # PyTorch model  
│   └── hindi_ocr_model.keras               # (optional TF fallback)  
├── fonts/                                  # Font files  
│   └── NotoSansDevanagari-Regular.ttf  
├── dataset/                                # Test data  
│   ├── images/                             # Input images for OCR  
│   │   └── training images                 # sample handwritten Hindi image  
│   └── words/                              # Expected‑output text files  
│       └── output labels                   # sample transcription  
├── label_encoder.pkl                       # sklearn LabelEncoder for class decoding 
│── Sample_OCR_Image.png                    # example image of OCR performace 
└── README.md                               # Project documentation  
</code></pre>

  <hr>

  <h2>Table of Contents</h2>
  <ul>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#configuration">Configuration</a></li>
    <li><a href="#running-the-api">Running the API</a></li>
    <li><a href="#api-endpoints">API Endpoints</a></li>
    <li><a href="#example-usage">Example Usage</a></li>
    <li><a href="#troubleshooting">Troubleshooting</a></li>
    <li><a href="#license">License</a></li>
  </ul>

  <hr>

  <h2 id="installation">Installation</h2>
  <ol>
    <li><strong>Clone repo</strong><br>
      <code>git clone https://github.com/Stu-ops/hindi-ocr.git<br>cd hindi-ocr</code>
    </li>
    <li><strong>Virtual environment</strong><br>
      <code>python -m venv venv<br>
      source venv/bin/activate   # macOS/Linux<br>
      venv\Scripts\activate      # Windows</code>
    </li>
    <li><strong>Install dependencies</strong><br>
      <code>pip install -r requirements.txt<br>
      pip install python-multipart    # for file uploads</code>
    </li>
  </ol>

  <hr>

  <h2 id="configuration">Configuration</h2>
  <p>Ensure the following files/folders sit next to <code>app.py</code>:</p>
  <ul>
    <li><code>models/</code> (your <code>.keras</code> and optional <code>.pth</code> weights)</li>
    <li><code>label_encoder.pkl</code> (sklearn LabelEncoder)</li>
    <li><code>fonts/NotoSansDevanagari-Regular.ttf</code></li>
    <li><code>dataset/</code> (training and testing images, words for testing)</li>
  </ul>

  <hr>

  <h2 id="running-the-api">Running the API</h2>
  <pre><code>uvicorn app:app --reload --host 0.0.0.0 --port 8000</code></pre>
  <p>
    <strong>Swagger UI</strong>: <a href="http://localhost:8000/docs">/docs</a><br>
    <strong>ReDoc</strong>: <a href="http://localhost:8000/redoc">/redoc</a>
  </p>

  <hr>

  <h2 id="api-endpoints">API Endpoints</h2>
  <table>
    <tr><th>Method</th><th>Path</th><th>Description</th></tr>
    <tr><td>GET</td><td><code>/</code></td><td>Welcome HTML page</td></tr>
    <tr><td>POST</td><td><code>/process/</code></td><td>Upload image → returns OCR &amp; prediction</td></tr>
    <tr><td>GET</td><td><code>/word-detection/</code></td><td>Returns word‑boxed image</td></tr>
    <tr><td>GET</td><td><code>/prediction/</code></td><td>Returns prediction‑overlay image</td></tr>
  </table>

  <h3>Response Schema for <code>/process/</code></h3>
  <pre><code>{
  "OCR_output": "यह एक उदाहरण है",
  "word_count": 5,
  "prediction_label": "अ"
}
  </code></pre>

  <hr>

  <h2 id="example-usage">Example Usage</h2>
  <h4>cURL</h4>
  <pre><code>curl -X POST "http://localhost:8000/process/" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@dataset/example1.png"</code></pre>

  <h4>Python</h4>
  <pre><code>import requests

url = "http://localhost:8000/process/"
with open("dataset/example1.png","rb") as f:
    files = {"file": f}
    resp = requests.post(url, files=files)
print(resp.json())</code></pre>

  <p>Retrieve word‑detection image:</p>
  <pre><code>curl http://localhost:8000/word-detection/ --output words.png</code></pre>

  <hr>

  <h2 id="troubleshooting">Troubleshooting</h2>
  <ul>
    <li><strong>“Form data requires python-multipart”</strong> → <code>pip install python-multipart</code></li>
    <li><strong>PermissionError on Windows</strong> → ensure you close image files before deletion; use <code>with Image.open(...)</code></li>
    <li><strong>Missing glyph warnings</strong> → add fallback font:<br>
      <code>plt.rcParams['font.family'] = ['Noto Sans Devanagari', 'DejaVu Sans']</code>
    </li>
  </ul>

  <h2 id="license">License</h2>
  <p>MIT © Stu-ops</p>

</body>
</html>
