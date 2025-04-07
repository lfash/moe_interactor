import os
import sys
import argparse
import re
import numpy as np
import faiss
import pickle
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Function to clean text of problematic characters
def clean_text(text):
    if not text:
        return ""
    
    # Replace any non-ascii characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# Function to load segments from text files
def load_segments(segments_dir):
    segments = []
    segment_files = [f for f in os.listdir(segments_dir) if f.endswith('.txt')]
    
    print(f"Loading {len(segment_files)} segment files...")
    for filename in tqdm(segment_files):
        filepath = os.path.join(segments_dir, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            
            # Extract metadata from file headers
            metadata = {}
            content_start = 0
            
            for i, line in enumerate(lines):
                line = clean_text(line)  # Clean each line
                if line.startswith("Title:"):
                    metadata["title"] = line.replace("Title:", "").strip()
                elif line.startswith("Episode:"):
                    # Store the original episode code
                    metadata["session_code"] = line.replace("Episode:", "").strip()
                    
                    # Try to extract session type and number
                    session_parts = metadata["session_code"].split('_')
                    session_type = ""
                    
                    if len(session_parts) > 0:
                        if session_parts[0] == "LS":
                            session_type = "Live Session"
                            if len(session_parts) >= 2:
                                try:
                                    metadata["session_number"] = int(session_parts[1])
                                except:
                                    metadata["session_number"] = 0
                        elif session_parts[0] == "PER":
                            session_type = "Personal Track"
                            if len(session_parts) >= 2:
                                try:
                                    metadata["module_number"] = int(session_parts[1])
                                except:
                                    metadata["module_number"] = 0
                        elif session_parts[0] == "PRO":
                            session_type = "Professional Track"
                            if len(session_parts) >= 2:
                                try:
                                    metadata["module_number"] = int(session_parts[1])
                                except:
                                    metadata["module_number"] = 0
                        elif session_parts[0] == "VC":
                            session_type = "Voice Clip"
                    
                    metadata["session_type"] = session_type
                    
                elif line.startswith("Segment:"):
                    try:
                        metadata["segment_number"] = int(line.replace("Segment:", "").strip())
                    except:
                        metadata["segment_number"] = 0
                
                # Find where the content starts (after the blank line)
                if line.strip() == "":
                    content_start = i + 1
                    break
            
            # Extract content and clean it
            content = "".join(lines[content_start:]).strip()
            content = clean_text(content)
            
            # Generate a segment ID based on the original session code
            segment_id = f"{metadata.get('session_code', 'unknown')}_{metadata.get('segment_number', '0')}"
            
            # Create segment object
            segment = {
                "content": content,
                "session_title": metadata.get("session_code", "Unknown Session"),
                "title": metadata.get("title", "Untitled Segment"),
                "session_type": metadata.get("session_type", "Unknown Type"),
                "session_number": metadata.get("session_number", 0),
                "module_number": metadata.get("module_number", 0),
                "segment_number": metadata.get("segment_number", 0),
                "segment_id": segment_id
            }
            
            segments.append(segment)
            
        except Exception as e:
            print(f"Error loading {filepath}: {str(e)}")
    
    return segments

# Function to create TF-IDF vectors
def create_tfidf_vectors(segments):
    print("Creating TF-IDF vectors...")
    
    # Extract processed content
    texts = [segment["content"] for segment in segments]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=1536, stop_words='english')  # Use 1536 features to match OpenAI's dimension
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Convert to dense vectors
    tfidf_vectors = tfidf_matrix.toarray()
    
    return tfidf_vectors, vectorizer

# Function to create and save FAISS index
def create_faiss_index(vectors, segments, output_dir, vectorizer):
    print("Creating FAISS index...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create FAISS index
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    # Add vectors to index
    index = faiss.IndexIDMap(index)
    index.add_with_ids(
        np.array(vectors).astype('float32'),
        np.array(range(len(segments))).astype('int64')
    )
    
    # Save index
    faiss.write_index(index, os.path.join(output_dir, "course_segments.index"))
    
    # Save vectorizer
    with open(os.path.join(output_dir, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    
    # Save segments metadata
    with open(os.path.join(output_dir, "segments_metadata.json"), "w", encoding="utf-8") as f:
        # Convert the segments to a format that can be easily serialized to JSON
        serializable_segments = []
        for segment in segments:
            serializable_segments.append(segment)
        
        json.dump(serializable_segments, f, ensure_ascii=True, indent=2)
    
    # Save a smaller pickle version as backup
    with open(os.path.join(output_dir, "segments_metadata.pkl"), "wb") as f:
        pickle.dump(segments, f)
    
    print(f"Index and metadata saved to {output_dir}")
    print(f"Total segments indexed: {len(segments)}")

# Main function
def main(segments_dir, output_dir="vector_db"):
    # Check if segments directory exists
    if not os.path.exists(segments_dir):
        print(f"Segments directory not found: {segments_dir}")
        return
    
    # Load segments
    segments = load_segments(segments_dir)
    print(f"Loaded {len(segments)} segments")
    
    if len(segments) == 0:
        print("No segments found. Exiting.")
        return
        
    # Create TF-IDF vectors
    vectors, vectorizer = create_tfidf_vectors(segments)
    
    # Create and save FAISS index
    create_faiss_index(vectors, segments, output_dir, vectorizer)
    
    # Save model info for reference
    with open(os.path.join(output_dir, "model_info.txt"), "w") as f:
        f.write("Model: TF-IDF Vectorizer\n")
        f.write(f"Embedding dimension: {vectors.shape[1]}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create vector embeddings for MoE course segments")
    parser.add_argument("segments_dir", help="Directory containing segment text files")
    parser.add_argument("--output_dir", help="Directory to save the vector database", default="vector_db")
    args = parser.parse_args()
    
    main(args.segments_dir, args.output_dir)