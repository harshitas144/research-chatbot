from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import logging
import numpy as np
from typing import List, Dict, Any, Tuple
import random
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import uuid
import traceback
import re
from http import HTTPStatus
from werkzeug.wrappers import Request, Response

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, supports_credentials=True, methods=['GET', 'POST', 'OPTIONS'])

# Copy all functions, imports, and global variables from app.py
# For brevity, assume all code from app.py is included here
# Example: load_summaries, load_papers_and_topics, get_topics, etc.

# Add your app.py content here (routes, functions, etc.)
# Ensure paths to data files are correct
os.environ['SUMMARIES_PATH'] = os.path.join(os.path.dirname(__file__), '../Summaries.xlsx')
os.environ['TOPICS_PATH'] = os.path.join(os.path.dirname(__file__), '../papers_by_topic.txt')
os.environ['EMBEDDINGS_PATH'] = os.path.join(os.path.dirname(__file__), '../paper_embeddings.npy')

# Vercel serverless function handler
def handler(environ, start_response):
    request = Request(environ)
    response = app(request)
    return Response(
        response=response.get_data(),
        status=response.status_code,
        headers=dict(response.headers)
    )(environ, start_response)

# Load models at startup
load_models()
setup_gemini_api()
