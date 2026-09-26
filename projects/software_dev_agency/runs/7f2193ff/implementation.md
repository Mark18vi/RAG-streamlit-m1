```python
# Minimal implementation of the Health Insurance Onboarding Platform

from flask import Flask, request, jsonify, session
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# User data storage (for demonstration purposes)
users = {}

@app.route('/register', methods=['POST'])
def register():
    username = request.json.get('username')
    password = request.json.get('password')
    if username in users:
        return jsonify({"error": "User already exists"}), 400
    users[username] = password
    return jsonify({"message": "User registered successfully"}), 201

@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username')
    password = request.json.get('password')
    if users.get(username) == password:
        session['user'] = username
        return jsonify({"message": "Logged in successfully"}), 200
    return jsonify({"error": "Invalid credentials"}), 401

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not file.filename.endswith(('.pdf', '.jpg', '.png')):
        return jsonify({"error": "Unsupported file type"}), 400
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return jsonify({"message": "File uploaded successfully"}), 201

@app.route('/progress', methods=['GET'])
def progress():
    return jsonify({"progress": "Step 1 of 5"}), 200

@app.route('/faq', methods=['GET'])
def faq():
    faqs = [
        {"question": "What is health insurance?", "answer": "Health insurance is..."},
        {"question": "How to select a plan?", "answer": "You can..."}
    ]
    return jsonify(faqs), 200

if __name__ == '__main__':
    app.run(debug=True)
```

```python
# Tests for the Health Insurance Onboarding Platform

import unittest
from app import app

class OnboardingTests(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_register_user(self):
        response = self.app.post('/register', json={'username': 'testuser', 'password': 'testpass'})
        self.assertEqual(response.status_code, 201)

    def test_login_user(self):
        self.app.post('/register', json={'username': 'testuser', 'password': 'testpass'})
        response = self.app.post('/login', json={'username': 'testuser', 'password': 'testpass'})
        self.assertEqual(response.status_code, 200)

    def test_upload_file(self):
        with open('test_image.jpg', 'wb') as f:
            f.write(b'test data')
        with open('test_image.jpg', 'rb') as f:
            response = self.app.post('/upload', data={'file': f})
            self.assertEqual(response.status_code, 201)

    def test_faq_endpoint(self):
        response = self.app.get('/faq')
        self.assertEqual(response.status_code, 200)
        self.assertIn('What is health insurance?', str(response.data))

if __name__ == '__main__':
    unittest.main()
```