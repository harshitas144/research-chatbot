from flask import Flask, request, jsonify, make_response
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

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, supports_credentials=True, methods=['GET', 'POST', 'OPTIONS'])

# Add CORS headers
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Global variables
summaries_df = None
papers_df = None
doc_ids = None
embeddings = None
vectorizer = None
gemini_available = False
topic_names = {}  # Dictionary to store topic ID to name mapping

# Attempt to load optional dependencies
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.warning("dotenv not installed.")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google.generativeai not installed.")

try:
    from fuzzywuzzy import fuzz
except ImportError:
    logger.warning("fuzzywuzzy not installed. Falling back to basic string matching.")
    fuzz = None

def normalize_title(title: str) -> str:
    """Normalize a title for matching."""
    if not isinstance(title, str):
        return ''
    title = title.lower().strip()
    title = re.sub(r'\s+', ' ', title)
    title = re.sub(r'\s*:\s*', ': ', title)
    title = re.sub(r'[^\w\s:]', '', title)
    return title

def load_summaries():
    """Load summaries from summaries.xlsx."""
    global summaries_df
    try:
        summaries_path = os.environ.get('SUMMARIES_PATH', './Summaries.xlsx')
        if os.path.exists(summaries_path):
            summaries_df = pd.read_excel(summaries_path)
            required_columns = ['Title', 'Problem', 'Solution/Methodology', 'Central Findings', 'Future Research']
            missing_columns = [col for col in required_columns if col not in summaries_df.columns]
            if missing_columns:
                logger.error(f"summaries.xlsx missing required columns: {missing_columns}")
                raise ValueError(f"Missing columns: {missing_columns}")
            if 'original_index' not in summaries_df.columns:
                summaries_df['original_index'] = summaries_df.index
            summaries_df['Title_normalized'] = summaries_df['Title'].apply(normalize_title)
            logger.info(f"Loaded summaries from {summaries_path}. Titles: {summaries_df['Title'].tolist()}")
            return True
        else:
            logger.warning(f"summaries.xlsx not found at {summaries_path}")
            summaries_df = pd.DataFrame({
                'Title': [f"Sample Summary {i}" for i in range(1, 6)],
                'Title_normalized': [f"sample summary {i}" for i in range(1, 6)],
                'Problem': [f"Research problem description {i}" for i in range(1, 6)],
                'Solution/Methodology': [f"Method used to solve problem {i}" for i in range(1, 6)],
                'Central Findings': [f"Key findings from the research {i}" for i in range(1, 6)],
                'Future Research': [f"Future research directions {i}" for i in range(1, 6)],
                'original_index': list(range(5))
            })
            logger.info(f"Fell back to sample summaries. Titles: {summaries_df['Title'].tolist()}")
            return True
    except Exception as e:
        logger.error(f"Error loading summaries: {e}")
        summaries_df = pd.DataFrame({
            'Title': [f"Sample Summary {i}" for i in range(1, 6)],
            'Title_normalized': [f"sample summary {i}" for i in range(1, 6)],
            'Problem': [f"Research problem description {i}" for i in range(1, 6)],
            'Solution/Methodology': [f"Method used to solve problem {i}" for i in range(1, 6)],
            'Central Findings': [f"Key findings from the research {i}" for i in range(1, 6)],
            'Future Research': [f"Future research directions {i}" for i in range(1, 6)],
            'original_index': list(range(5))
        })
        logger.info(f"Fell back to sample summaries due to error. Titles: {summaries_df['Title'].tolist()}")
        return True

def load_papers_and_topics():
    """Load papers and topics from papers_by_topics.txt, including topic names."""
    global papers_df, topic_names
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_path = os.path.join(script_dir, 'papers_by_topic.txt')
        topics_path = os.environ.get('TOPICS_PATH', default_path)
        
        logger.info(f"Attempting to load papers_by_topics.txt from: {topics_path}")
        
        if not os.path.exists(topics_path):
            logger.warning(f"papers_by_topics.txt not found at {topics_path}")
            raise FileNotFoundError("File not found")

        papers_data = []
        current_topic_id = None
        current_topic_name = None

        with open(topics_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                if not line or line.startswith('-------'):
                    continue
                topic_match = re.match(r'## Topic (\d+): (.+)', line)
                if topic_match:
                    current_topic_id = int(topic_match.group(1))
                    current_topic_name = topic_match.group(2).strip()
                    topic_names[current_topic_id] = current_topic_name
                    logger.debug(f"Found topic: ID={current_topic_id}, Name={current_topic_name}")
                    continue
                paper_match = re.match(r'-\s*(.+)', line)
                if paper_match and current_topic_id is not None:
                    paper_title = paper_match.group(1).strip()
                    papers_data.append({
                        'Title': paper_title,
                        'Title_normalized': normalize_title(paper_title),
                        'Topic': current_topic_id,
                        'Topic_Name': current_topic_name
                    })
                    logger.debug(f"Found paper: {paper_title} under topic {current_topic_name}")
                    continue

        if not papers_data:
            logger.warning("No papers found in papers_by_topics.txt")
            raise ValueError("No papers parsed")

        papers_df = pd.DataFrame(papers_data)
        papers_df['original_index'] = papers_df.index
        logger.info(f"Loaded {len(papers_df)} papers and topics from {topics_path}. Topics: {list(topic_names.values())}")
        return True

    except Exception as e:
        logger.error(f"Error loading papers and topics: {e}")
        papers_df = pd.DataFrame({
            'Title': [f"Sample Paper {i}" for i in range(1, 6)],
            'Title_normalized': [f"sample paper {i}" for i in range(1, 6)],
            'Topic': [i % 2 for i in range(5)],
            'Topic_Name': [f"Sample Topic {i % 2 + 1}" for i in range(5)],
            'original_index': list(range(5))
        })
        topic_names = {0: "Sample Topic 1", 1: "Sample Topic 2"}
        logger.info(f"Fell back to sample papers. Topics: {list(topic_names.values())}")
        return True

def create_simple_embeddings(df):
    """Create TF-IDF embeddings for summaries."""
    global doc_ids, embeddings, vectorizer
    try:
        text_data = []
        doc_ids = []
        for idx, row in df.iterrows():
            combined_text = f"{row.get('Title', '')} {row.get('Title', '')} "
            for field in ['Problem', 'Solution/Methodology', 'Central Findings', 'Future Research']:
                if field in row and pd.notna(row[field]):
                    combined_text += f"{row[field]} "
            if combined_text.strip():
                text_data.append(combined_text.strip().lower())
                doc_ids.append(idx)
        
        vectorizer = TfidfVectorizer(max_features=100)
        embeddings = vectorizer.fit_transform(text_data).toarray()
        return True
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        embeddings = np.random.rand(len(text_data), 50)
        return True

def load_models():
    """Load summaries, papers, and embeddings."""
    load_summaries()
    load_papers_and_topics()
    embeddings_loaded = False
    embeddings_path = os.environ.get('EMBEDDINGS_PATH', './paper_embeddings.npy')
    if os.path.exists(embeddings_path):
        try:
            embeddings = np.load(embeddings_path)
            doc_ids = list(range(len(embeddings)))
            embeddings_loaded = True
        except Exception:
            pass
    if not embeddings_loaded and summaries_df is not None:
        embeddings_loaded = create_simple_embeddings(summaries_df)
        if embeddings_loaded:
            try:
                np.save(embeddings_path, embeddings)
            except Exception:
                pass
    return True

def setup_gemini_api():
    """Setup Gemini API with updated model selection."""
    global gemini_available
    if not GEMINI_AVAILABLE:
        logger.warning("Gemini API package not installed.")
        return False
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        logger.warning("GEMINI_API_KEY not set.")
        return False
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        model_names = [model.name for model in models]
        logger.info(f"Available Gemini models: {model_names}")
        target_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
        selected_model = None
        for model_name in target_models:
            matching_model = next((name for name in model_names if model_name in name), None)
            if matching_model:
                selected_model = matching_model
                break
        if selected_model:
            logger.info(f"Selected Gemini model: {selected_model}")
            model = genai.GenerativeModel(model_name=selected_model)
            response = model.generate_content("Hello")
            gemini_available = True
            os.environ['GEMINI_MODEL'] = selected_model
            return True
        else:
            logger.error("No suitable Gemini model found.")
            return False
    except Exception as e:
        logger.error(f"Error configuring Gemini API: {e}")
        return False

def get_topic_info() -> List[Dict[str, Any]]:
    """Get topic information from papers_df with actual topic names."""
    if papers_df is not None and 'Topic' in papers_df.columns:
        topics = papers_df['Topic'].unique()
        topic_list = []
        for topic_id in sorted(topics):
            if topic_id == -1:
                continue
            paper_count = len(papers_df[papers_df['Topic'] == topic_id])
            topic_papers = papers_df[papers_df['Topic'] == topic_id]
            all_title_words = ' '.join(topic_papers['Title'].astype(str)).lower().split()
            word_counts = Counter(all_title_words)
            stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'of', 'and', 'or', 'to', 'for'}
            keywords = [word for word, count in word_counts.most_common(10) if word not in stop_words and len(word) > 2][:5]
            topic_name = topic_names.get(topic_id, f"Topic {topic_id + 1}")
            topic_list.append({
                'id': int(topic_id),
                'name': topic_name,
                'count': int(paper_count),
                'keywords': keywords or ['research', 'paper', 'topic', f'keyword{topic_id}']
            })
        return topic_list
    return [
        {'id': 0, 'name': 'Sample Topic 1', 'count': 5, 'keywords': ['research', 'papers', 'analysis']},
        {'id': 1, 'name': 'Sample Topic 2', 'count': 3, 'keywords': ['method', 'results', 'study']}
    ]

def get_papers_by_topic(topic_id: int) -> List[Dict[str, Any]]:
    """Get papers by topic with actual topic name."""
    if papers_df is None:
        return []
    topic_papers = papers_df[papers_df['Topic'] == topic_id]
    topic_name = topic_names.get(topic_id, f"Topic {topic_id + 1}")
    papers_list = []
    for idx, paper in topic_papers.iterrows():
        paper_dict = {
            'id': int(paper.get('original_index', idx)),
            'title': paper.get('Title', f"Paper #{idx}"),
            'topic': int(topic_id),
            'topic_name': topic_name
        }
        summary = get_summary_by_title(paper.get('Title', ''))
        if summary:
            paper_dict['summary'] = summary.get('summary', '')
            for field in ['Problem', 'Solution/Methodology', 'Central Findings', 'Future Research']:
                if field in summary:
                    field_key = field.lower().replace('/', '_').replace(' ', '_')
                    paper_dict[field_key] = summary[field]
        papers_list.append(paper_dict)
    return papers_list

def get_paper_details(paper_id: int) -> Dict[str, Any]:
    """Get paper details by ID from summaries_df with topic name."""
    if summaries_df is None:
        logger.warning("summaries_df is None. Cannot retrieve paper details.")
        return {}
    try:
        paper_row = summaries_df[summaries_df['original_index'] == paper_id]
        paper = paper_row.iloc[0] if len(paper_row) > 0 else None
        if paper is None:
            logger.warning(f"No paper found with ID {paper_id}.")
            return {}
        # Find topic from papers_df if available
        topic_id = -1
        topic_name = "N/A"
        if papers_df is not None:
            matching_papers = papers_df[papers_df['Title_normalized'] == normalize_title(paper.get('Title', ''))]
            if not matching_papers.empty:
                topic_id = int(matching_papers.iloc[0]['Topic'])
                topic_name = topic_names.get(topic_id, f"Topic {topic_id + 1}")
        paper_dict = {
            'id': paper_id,
            'title': paper.get('Title', f"Paper #{paper_id}"),
            'topic': topic_id,
            'topic_name': topic_name
        }
        for field in ['Title', 'Problem', 'Solution/Methodology', 'Central Findings', 'Future Research']:
            if field in paper and isinstance(paper[field], str) and len(str(paper[field]).strip()) > 0:
                field_key = field.lower().replace('/', '_').replace(' ', '_')
                paper_dict[field_key] = str(paper[field])
            else:
                field_key = field.lower().replace('/', '_').replace(' ', '_')
                paper_dict[field_key] = "N/A"
        return paper_dict
    except Exception as e:
        logger.error(f"Error retrieving paper details for ID {paper_id}: {e}")
        return {}

def get_summary_by_title(title: str) -> Dict[str, Any]:
    """Get summary by title with improved fuzzy matching from summaries.xlsx."""
    if summaries_df is None:
        logger.warning("summaries_df is None. Cannot retrieve summary.")
        return {}
    try:
        title_clean = normalize_title(title)
        best_match = None
        best_score = 0.0
        best_index = -1

        for idx, row in summaries_df.iterrows():
            summary_title = row['Title_normalized']
            if not summary_title:
                continue

            if fuzz:
                score = fuzz.token_sort_ratio(title_clean, summary_title) / 100.0
                logger.debug(f"Comparing summary titles: '{title_clean}' vs '{summary_title}' (score: {score})")
            else:
                if title_clean in summary_title or summary_title in title_clean:
                    score = len(title_clean) / len(summary_title) if len(summary_title) > 0 else 0
                else:
                    score = 0.0

            threshold = 0.4 if fuzz else 0.3
            if score > best_score and score >= threshold:
                best_score = score
                best_match = row
                best_index = idx

        if best_match is not None:
            summary_id = int(best_match.get('original_index', best_index))
            logger.info(f"Found summary for title '{title}' (matched '{best_match.get('Title', '')}' with score {best_score})")
            result = {
                'id': summary_id,
                'title': best_match.get('Title', f"Summary #{summary_id}"),
                'summary': best_match.get('Summary', '')
            }
            for field in ['Problem', 'Solution/Methodology', 'Central Findings', 'Future Research']:
                if field in best_match and pd.notna(best_match[field]):
                    result[field] = str(best_match[field])
            return result
        else:
            logger.warning(f"No summary found for title '{title}'. Best score: {best_score}")
            return {}
    except Exception as e:
        logger.error(f"Error in get_summary_by_title for title '{title}': {e}")
        return {}

def get_paper_by_title(title: str) -> Dict[str, Any]:
    """Get paper by title from summaries.xlsx."""
    summary = get_summary_by_title(title)
    if summary:
        paper_id = summary.get('id', -1)
        return get_paper_details(paper_id)
    return {}

def query_gemini_api(query: str, context: str = "", web_search: bool = False) -> str:
    """Query Gemini API, optionally with web search context."""
    global gemini_available
    if not GEMINI_AVAILABLE or not gemini_available:
        return "Gemini API unavailable. I can provide paper summaries from the database."
    try:
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
        }
        model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
        prompt = f"{context}\n\n{query}"
        if web_search:
            prompt = f"Search the web for recent information about: {query}\nProvide a detailed summary based on the latest available data."
        response = model.generate_content(prompt)
        return response.text if response and hasattr(response, 'text') else "Incomplete response from Gemini API."
    except Exception as e:
        return f"Error with Gemini API: {str(e)}."

def find_relevant_papers(query: str, top_n: int = 3) -> List[Dict[str, Any]]:
    """Find relevant papers using embeddings."""
    if summaries_df is None or embeddings is None or doc_ids is None or vectorizer is None:
        sample_size = min(top_n, len(summaries_df)) if summaries_df is not None else 0
        random_papers = summaries_df.sample(sample_size) if sample_size > 0 else []
        return [get_paper_details(idx) for idx in random_papers.index]
    try:
        query_lower = query.lower()
        query_embedding = vectorizer.transform([query_lower]).toarray()[0]
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        relevant_papers = []
        for idx in top_indices:
            if idx < len(doc_ids):
                paper_idx = doc_ids[idx]
                paper = get_paper_details(paper_idx)
                if paper:
                    paper['relevance'] = float(similarities[idx])
                    relevant_papers.append(paper)
        return relevant_papers
    except Exception:
        sample_size = min(top_n, len(summaries_df)) if summaries_df is not None else 0
        random_papers = summaries_df.sample(sample_size) if sample_size > 0 else []
        return [get_paper_details(idx) for idx in random_papers.index]

def extract_paper_title(query: str) -> str:
    """Extract paper title from query with improved parsing."""
    query_clean = query.strip()
    query_lower = query_clean.lower()
    
    title_indicators = [
        'paper titled', 'paper called', 'paper on', 'about the paper',
        'details of', 'info on', 'information about', 'paper about',
        'titled', 'called', 'named', 'summary of'
    ]
    
    for indicator in title_indicators:
        if indicator in query_lower:
            start_idx = query_lower.find(indicator) + len(indicator)
            title = query_clean[start_idx:].strip()
            stop_phrases = [' by ', ' from ', ' in ', ' on ', ' at ', ' regarding ']
            for phrase in stop_phrases:
                if phrase in title.lower():
                    title = title[:title.lower().find(phrase)].strip()
            title = title.strip('"\'')
            if title:
                logger.info(f"Extracted title using indicator '{indicator}': {title}")
                return title

    quote_match = re.search(r'[\'"](.+?)[\'"]', query_clean)
    if quote_match:
        title = quote_match.group(1).strip()
        logger.info(f"Extracted quoted title: {title}")
        return title

    question_words = {'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'can', 'find', 'search'}
    query_words = set(query_lower.split())
    if not question_words.intersection(query_words):
        logger.info(f"Using entire query as title: {query_clean}")
        return query_clean

    logger.info("No specific title extracted, returning empty string")
    return ""

def is_paper_detail_query(query: str) -> Tuple[bool, str, bool]:
    """Check if query is asking for specific paper details."""
    query_lower = query.lower()
    detail_keywords = [
        'paper details', 'paper info', 'paper information', 'specific paper',
        'paper id', 'about paper', 'detail of paper', 'explain about', 'tell me about',
        'the paper titled', 'info on', 'summary of', 'details of'
    ]
    more_info_keywords = [
        'tell me more', 'explain more', 'explain further', 'more about',
        'more details', 'deeper insight', 'in depth'
    ]
    has_id = any(word.isdigit() for word in query_lower.split())
    is_detail = has_id or any(keyword in query_lower for keyword in detail_keywords)
    wants_more = any(keyword in query_lower for keyword in more_info_keywords)
    
    paper_title = extract_paper_title(query)
    logger.info(f"Query analysis: is_detail={is_detail}, extracted_title='{paper_title}', wants_more={wants_more}")
    
    return is_detail, paper_title, wants_more

def is_relevant_papers_query(query: str) -> bool:
    """Check if query is asking for relevant papers."""
    query_lower = query.lower()
    relevant_keywords = [
        'relevant papers', 'related papers', 'similar papers', 'find papers',
        'papers on', 'papers about', 'search for papers', 'list of papers'
    ]
    return any(keyword in query_lower for keyword in relevant_keywords)

def extract_paper_id(query: str) -> int:
    """Extract paper ID from query."""
    query_lower = query.lower()
    id_match = re.search(r'paper\s*(?:id\s*)?(\d+)', query_lower)
    if id_match:
        return int(id_match.group(1))
    for word in query_lower.split():
        if word.isdigit():
            return int(word)
    return -1

def generate_fallback_response(query: str, relevant_papers: List[Dict[str, Any]] = None) -> str:
    """Generate a fallback response."""
    if relevant_papers:
        response = f"Found {len(relevant_papers)} relevant papers:\n\n"
        for i, paper in enumerate(relevant_papers[:3]):
            title = paper.get('title', f"Paper #{i+1}")
            problem = paper.get('problem', paper.get('central_findings', ''))
            topic_name = paper.get('topic_name', 'N/A')
            if problem and len(problem) > 500:
                problem = problem[:497] + "..."
            response += f"• {title}\n  Problem: {problem or 'N/A'}\n  Topic: {topic_name}\n"
        response += "\nAsk about a specific paper for more details."
        return response
    return "I couldn't find any relevant information for your query."

@app.route('/api/topics', methods=['GET'])
def get_topics():
    try:
        return jsonify({'topics': get_topic_info()})
    except Exception as e:
        return jsonify({'error': str(e), 'topics': []}), 500

@app.route('/api/papers', methods=['GET'])
def get_papers():
    try:
        topic_id = request.args.get('topic_id', type=int)
        papers = get_papers_by_topic(topic_id) if topic_id is not None else [get_paper_details(idx) for idx in summaries_df.head(100).index] if summaries_df is not None else []
        return jsonify({'papers': papers})
    except Exception as e:
        return jsonify({'error': str(e), 'papers': []}), 500

@app.route('/api/paper/<int:paper_id>', methods=['GET'])
def get_paper(paper_id):
    try:
        paper = get_paper_details(paper_id)
        if not paper:
            return jsonify({'error': 'Paper not found'}), 404
        return jsonify({'paper': paper})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint with improved query handling."""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing query parameter'}), 400
        
        query = data['query']
        is_detail_query, paper_title, wants_more_info = is_paper_detail_query(query)
        is_relevant_query = is_relevant_papers_query(query)
        relevant_papers = []
        debug_info = ""

        if is_detail_query:
            paper = None
            paper_id = extract_paper_id(query)
            if paper_id >= 0:
                paper = get_paper_details(paper_id)
                if paper:
                    logger.info(f"Found paper by ID {paper_id}: {paper.get('title', 'N/A')}")
                else:
                    debug_info = f"Paper with ID {paper_id} not found."

            if not paper and paper_title:
                paper = get_paper_by_title(paper_title)
                if paper:
                    logger.info(f"Found paper by title: {paper.get('title', 'N/A')}")
                else:
                    debug_info = f"No paper found for title '{paper_title}' in summaries.xlsx."

            if not paper and not paper_title and paper_id < 0:
                debug_info = "No paper ID or title provided in the query."

            if paper:
                markdown_summary = f"# {paper.get('title', 'N/A')}\n\n" \
                                 f"## Problem\n{paper.get('problem', 'N/A')}\n\n" \
                                 f"## Solution/Methodology\n{paper.get('solution_methodology', 'N/A')}\n\n" \
                                 f"## Central Findings\n{paper.get('central_findings', 'N/A')}\n\n" \
                                 f"## Future Research\n{paper.get('future_research', 'N/A')}\n\n" \
                                 f"## Topic\n{paper.get('topic_name', 'N/A')}"
                
                if wants_more_info and gemini_available:
                    context = f"Based on the following paper summary:\n\n{markdown_summary}\n\nProvide a detailed explanation or additional insights."
                    more_info = query_gemini_api(query, context)
                    markdown_summary += f"\n\n## Additional Insights\n{more_info}"
                
                response_text = markdown_summary
            else:
                response_text = "Paper not found in summaries.xlsx. Try specifying the title or ID more clearly."
                relevant_papers = find_relevant_papers(query, top_n=3)
                if relevant_papers:
                    response_text += "\n\nHere are some related papers that might help:\n"
                    for paper in relevant_papers:
                        response_text += f"- {paper.get('title', 'N/A')} (Topic: {paper.get('topic_name', 'N/A')})\n"

            return jsonify({
                'response': response_text,
                'relevant_papers': relevant_papers,
                'timestamp': datetime.datetime.now().isoformat(),
                'debug_info': debug_info
            })

        if is_relevant_query:
            relevant_papers = find_relevant_papers(query, top_n=5)
            response_text = generate_fallback_response(query, relevant_papers)
            return jsonify({
                'response': response_text,
                'relevant_papers': relevant_papers,
                'timestamp': datetime.datetime.now().isoformat(),
                'debug_info': debug_info
            })

        context = ""
        if paper_title:
            paper = get_paper_by_title(paper_title)
            if paper:
                response_text = f"# {paper.get('title', 'N/A')}\n\n" \
                              f"## Problem\n{paper.get('problem', 'N/A')}\n\n" \
                              f"## Solution/Methodology\n{paper.get('solution_methodology', 'N/A')}\n\n" \
                              f"## Central Findings\n{paper.get('central_findings', 'N/A')}\n\n" \
                              f"## Future Research\n{paper.get('future_research', 'N/A')}\n\n" \
                              f"## Topic\n{paper.get('topic_name', 'N/A')}"
            else:
                response_text = query_gemini_api(query, context, web_search=True) if gemini_available else "I can provide information from the database. Please ask about a specific paper or request relevant papers."
        else:
            response_text = query_gemini_api(query, context, web_search=True) if gemini_available else "I can provide information from the database. Please ask about a specific paper or request relevant papers."

        relevant_papers = find_relevant_papers(query, top_n=3)
        return jsonify({
            'response': response_text,
            'relevant_papers': relevant_papers,
            'timestamp': datetime.datetime.now().isoformat(),
            'debug_info': debug_info
        })

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'error': str(e),
            'response': "Error processing request.",
            'relevant_papers': [],
            'debug_info': 'Internal server error'
        }), 500

@app.route('/api/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing query parameter'}), 400
        query = data['query']
        num_results = max(1, min(data.get('num_results', 10), 50))
        relevant_papers = find_relevant_papers(query, top_n=num_results)
        return jsonify({
            'query': query,
            'papers': relevant_papers,
            'count': len(relevant_papers),
            'timestamp': datetime.datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'papers': [],
            'count': 0
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat(),
        'summaries_loaded': summaries_df is not None,
        'papers_loaded': papers_df is not None,
        'embeddings_loaded': embeddings is not None,
        'gemini_available': gemini_available
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    load_models()
    setup_gemini_api()
    app.run(host='0.0.0.0', port=port, debug=True)