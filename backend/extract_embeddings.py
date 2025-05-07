import pickle
import numpy as np
import os
import pandas as pd
from sentence_transformers import SentenceTransformer

def load_model(model_path):
    """Load the BERTopic model from pickle file."""
    try:
        with open(model_path, 'rb') as f:
            topic_model = pickle.load(f)
        print(f"Successfully loaded model from {model_path}")
        return topic_model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_data(data_path):
    """Load the papers DataFrame."""
    try:
        df = pd.read_excel(data_path) if data_path.endswith('.xlsx') else pd.read_csv(data_path)
        print(f"Successfully loaded {len(df)} papers from {data_path}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def prepare_text_data(df):
    """Combine relevant columns into a single text field for embedding."""
    # Normalize column names
    df_columns = df.columns.tolist()
    column_map = {}

    search_columns = ['Title', 'Problem', 'Solution/Methodolgy', 'Central findings', 'Future reserach']
    for target_col in search_columns:
        for actual_col in df_columns:
            if actual_col.lower() == target_col.lower() or actual_col.strip().lower() == target_col.strip().lower():
                column_map[target_col] = actual_col

    print(f"Using columns for text preparation: {list(column_map.values())}")

    # Prepare a list to store the combined text
    text_data = []
    doc_ids = []  # Store original dataframe indices

    for idx, row in df.iterrows():
        combined_text = ""

        # Always include title if available (highest priority)
        title_col = column_map.get('Title')
        if title_col and pd.notna(row.get(title_col, np.nan)):
            combined_text += str(row[title_col]) + " " + str(row[title_col]) + " "

        # Add other available columns
        for target_col, actual_col in column_map.items():
            if target_col != 'Title' and pd.notna(row.get(actual_col, np.nan)):
                combined_text += str(row[actual_col]) + " "

        # Include all papers - if no content was extracted, use a placeholder with the index
        if combined_text.strip():
            text_data.append(combined_text.strip())
        else:
            placeholder = f"Paper {idx} with no content data"
            text_data.append(placeholder)

        doc_ids.append(idx)

    return text_data, doc_ids

def extract_embeddings(model, text_data, output_file):
    """Extract embeddings using the model's embedding function and save to file."""
    try:
        # Get the embedding model from BERTopic
        embedding_model = model._embedding_model
        
        print(f"Extracting embeddings for {len(text_data)} papers...")
        embeddings = embedding_model.encode(text_data, show_progress_bar=True)
        
        # Save embeddings to file
        np.save(output_file, embeddings)
        print(f"Saved embeddings with shape {embeddings.shape} to {output_file}")
        return embeddings
    except Exception as e:
        print(f"Error extracting embeddings: {e}")
        return None

def main():
    # Configuration
    model_path = os.environ.get('MODEL_PATH', './bertopic_model_improved.pkl')
    data_path = os.environ.get('DATA_PATH', './Summaries.xlsx')
    output_file = os.environ.get('EMBEDDINGS_PATH', './paper_embeddings.npy')
    
    # Load model
    topic_model = load_model(model_path)
    if topic_model is None:
        return
    
    # Load data
    df = load_data(data_path)
    if df is None:
        return
    
    # Prepare text data
    text_data, doc_ids = prepare_text_data(df)
    print(f"Prepared {len(text_data)} documents for embedding extraction")
    
    # Extract and save embeddings
    embeddings = extract_embeddings(topic_model, text_data, output_file)
    
    if embeddings is not None:
        print("Embedding extraction completed successfully!")

if __name__ == "__main__":
    main()