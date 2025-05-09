import os
import streamlit as st
import anthropic
import faiss
import numpy as np
import json
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Understanding Foundations",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource(show_spinner=False)
def load_faiss_resources(vectordb_dir="vectordb"):
    # Get the absolute path to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct absolute path to vector_db
    vectordb_path = os.path.join(script_dir, vectordb_dir)
    
    # Debug output
    st.sidebar.write(f"Looking for files in: {vectordb_path}")
    
    index_path = os.path.join(vectordb_path, "course_segments.index")
    metadata_path = os.path.join(vectordb_path, "segments_metadata.json")
    vectorizer_path = os.path.join(vectordb_path, "vectorizer.pkl")
    
    # Check if each file exists
    st.sidebar.write(f"index_path exists: {os.path.exists(index_path)}")
    st.sidebar.write(f"metadata_path exists: {os.path.exists(metadata_path)}")
    st.sidebar.write(f"vectorizer_path exists: {os.path.exists(vectorizer_path)}")
    
    if not os.path.exists(index_path) or not os.path.exists(metadata_path) or not os.path.exists(vectorizer_path):
        st.error(f"Vector database files not found in {vectordb_path}")
        return None, None, None

# Check various possible locations
possible_paths = [
    "vectordb",
    "./vectordb",
    "../vectordb",
    os.path.join(os.getcwd(), "vectordb")
]

for path in possible_paths:
    st.sidebar.write(f"Checking {path} exists: {os.path.exists(path)}")
    if os.path.exists(path):
        st.sidebar.write(f"Files in {path}: {os.listdir(path)}")

# Custom CSS for UI styling
st.markdown(
    """
    <style>
    /* Overall styling */
    body { color: white; background-color: #191B1F; }
    .stApp { background-color: #191B1F; }
    
    /* Completely override Streamlit's button hover behavior */
    .stApp button:hover,
    .stApp [data-baseweb="button"]:hover,
    div[data-testid="stFormSubmitButton"] button:hover,
    div[data-testid="stFormSubmitButton"] [data-baseweb="button"]:hover,
    .stButton > button:hover,
    button:hover {
        background-color: #3b8a9d !important;
        border-color: #3b8a9d !important;
        color: white !important;
    }
    
    /* Force important on all button hover states */
    button:hover, 
    *[role="button"]:hover {
        background-color: #3b8a9d !important;
        border-color: #3b8a9d !important;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    section[data-testid="stSidebar"] {display: none;}
    div[data-baseweb="spinner"] {display: none !important;}
    
    /* Header styling - STICKY header */
    .app-header {
        background-color: #191B1F;
        padding: 1rem;
        border-bottom: 1px solid #317485;
        position: sticky;
        top: 0;
        z-index: 999;
    }
    .app-title {
    font-family: 'Futura', 'Trebuchet MS', sans-serif !important;
    color: white !important;
    font-size: 1.5rem !important;
    font-weight: 300 !important; /* Changed from normal to 300 for lighter weight */
    margin-bottom: 0.2rem !important;
    text-transform: none !important;
    letter-spacing: normal !important;
}
.app-subtitle {
    font-family: 'Futura', 'Trebuchet MS', sans-serif !important;
    color: #ccc !important;
    font-size: 0.9rem !important;
    font-weight: 300 !important;
    margin-top: 0 !important;
    text-transform: none !important;
    letter-spacing: normal !important;
}
    
    /* Message styling */
    .message-container { margin-bottom: 1.5rem; font-family: Arial, sans-serif; }
    .user-message {
        background-color: #2A2D36;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        color: white;
    }
    .assistant-message {
        background-color: #1E2026;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        color: white;
        border-left: 3px solid #317485;
    }
    
    /* Add paragraph spacing in assistant messages */
    .assistant-message p {
        margin-bottom: 1rem;
    }
    
    /* Ensure the last paragraph doesn't have extra margin */
    .assistant-message p:last-child {
        margin-bottom: 0;
    }
    
    /* Citation styling */
    .numbered-ref { 
        display: block;
        margin-top: 0.5rem;
        line-height: 1.4;
    }
    
    /* Form styling */
    .initial-prompt {
        color: white !important;
        font-family: 'Futura', 'Trebuchet MS', sans-serif !important;
        font-size: 1.3rem !important;
        font-weight: 300 !important;
        text-align: center !important;
        margin-bottom: 1rem !important;
    }
    .input-container {
        display: flex;
        align-items: center;
        position: relative;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        background-color: rgba(255, 255, 255, 0.05);
        width: 100%;
        margin-bottom: 0.5rem;
    }
    .stTextInput, .stForm > div[data-baseweb="form-control"] {
        flex-grow: 1;
        margin-bottom: 0 !important;
    }
    .stTextInput > div {
        border: none !important;
        background-color: transparent !important;
    }
    .stTextInput > div > div > input {
        background-color: #191B1F !important;  /* Match background color */
        color: white !important;
        font-family: Arial, sans-serif !important;
        border: none !important;
        padding: 0.75rem 1rem !important;
    }
    /* Remove box around form */
    .stForm > div:first-child {
        border: none !important;
        padding: 0 !important;
    }
    .stForm [data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
    }
    /* Button styles - adjusted for better fit */
    button[kind="formSubmit"] {
        background-color: #317485 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.3rem 0.6rem !important;
        font-size: 0.9rem !important;
        font-family: 'Futura', 'Trebuchet MS', sans-serif !important;
        font-weight: 400 !important;
        white-space: nowrap !important;
        margin-top: 0 !important;
        height: auto !important;
        min-height: 0 !important;
        transition: background-color 0.3s !important;
    }
    
    /* Override Streamlit's loading state stylings too */
    .stApp [data-testid="stFormSubmitButton"] [data-baseweb="button"][aria-disabled="true"],
    .stApp button[disabled],
    .stApp [data-baseweb="button"][disabled] {
        background-color: #317485 !important;
        opacity: 0.7;
        color: white !important;
    }
    
    /* Follow-up form centering */
    .followup-form {
        display: flex;
        justify-content: center;
        width: 100%;
        max-width: 768px;
        margin: 0 auto;
    }
    
    .followup-form .stForm {
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 100%;
    }
    
    .followup-form .stForm > div[data-baseweb="form-control"] {
        width: 100%;
    }
    
    .followup-form button[kind="formSubmit"] {
        margin: 0.5rem auto !important;
        display: block !important;
    }
    
    /* Module highlight section */
    .module-highlight {
        background-color: #1E2026;
        border: 1px solid #317485;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        overflow: hidden;
    }
    .module-title {
        font-family: 'Futura', 'Trebuchet MS', sans-serif;
        color: white;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    /* Loading indicator */
    .loading-indicator {
        color: #317485;
        font-style: italic;
        margin: 1rem 0;
    }
    
    /* New chat button */
    .new-chat-btn {
        color: white;
        background-color: transparent;
        border: 1px solid #317485;
        border-radius: 4px;
        padding: 0.25rem 0.75rem;
        font-size: 0.9rem;
        cursor: pointer;
        margin-top: 0.5rem;
        font-family: 'Futura', 'Trebuchet MS', sans-serif;
    }
    
    /* Empty state container */
    .empty-state-container {
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        align-items: center;
        padding-top: 100px; /* This will position it 100px from the top instead of centering vertically */
        width: 100%;
        max-width: 768px;
        margin: 0 auto;
    }
    
    /* Adjust column layout for form */
    .input-col {
        padding-right: 0 !important;
    }
    .button-col {
        padding-left: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Debug API key (only show length for security)
try:
    anthropic_api_key = st.secrets["ANTHROPIC_API_KEY"]
    st.sidebar.write("API key found, length:", len(anthropic_api_key))
    # Show first and last few characters to verify it's correct without revealing the full key
    st.sidebar.write("API key starts with:", anthropic_api_key[:7] + "..." + anthropic_api_key[-4:])
except Exception as e:
    st.sidebar.write("Error accessing API key:", str(e))
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Debug: Print directory information
st.sidebar.write("Current working directory:", os.getcwd())
st.sidebar.write("Files in directory:", os.listdir())
st.sidebar.write("Vectordb exists:", os.path.exists("vectordb"))
if os.path.exists("vectordb"):
    st.sidebar.write("Files in vectordb:", os.listdir("vectordb"))
else:
    st.sidebar.write("Vectordb directory not found!")

# Process text to format citations
def format_citations_html(text):
    # Find any numbered citations at the end of the text
    citation_pattern = r'(\d+\.\s+.+?(?:,\s*\d{4}))'
    citations = re.findall(citation_pattern, text.split("\n\n")[-1])
    
    # If citations found, replace them with formatted HTML
    if citations:
        formatted_citations = ""
        for citation in citations:
            formatted_citations += f'<span class="numbered-ref">{citation}</span>'
        
        # Remove the original citations and add the formatted ones
        text_parts = text.rsplit("\n\n", 1)
        if len(text_parts) > 1:
            text = text_parts[0] + "\n\n" + formatted_citations
        else:
            text = text + "\n\n" + formatted_citations
    
    return text

# Initialize API clients
try:
    # Try to get API key from secrets first, then fall back to env vars
    anthropic_api_key = st.secrets.get("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY"))
    anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
except Exception as e:
    st.error(f"Error connecting to Claude API: {e}")
    anthropic_client = None

# Function to clean text
def clean_text(text):
    if not text:
        return ""
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Name correction function
def correct_names(text):
    text = re.sub(r'\bSanjay\b', 'Bree', text, flags=re.IGNORECASE)
    return text

# Function to extract segment information for citation
def extract_segment_info(segment_id):
    parts = segment_id.split('_')
    
    if len(parts) < 2:
        return None
    
    segment_type = None
    module_num = None
    segment_num = None
    
    if parts[0] == "LS":
        segment_type = "Live Session"
        try:
            module_num = parts[1]
        except IndexError:
            pass
    elif parts[0] == "PER":
        segment_type = "Personal Track"
        try:
            module_raw = parts[1]
            if "Module" in module_raw:
                module_num = module_raw.replace("Module", "")
            else:
                module_num = module_raw
        except IndexError:
            pass
    elif parts[0] == "PRO":
        segment_type = "Professional Track"
        try:
            module_raw = parts[1]
            if "Module" in module_raw:
                module_num = module_raw.replace("Module", "")
            else:
                module_num = module_raw
        except IndexError:
            pass
    
    try:
        if len(parts) > 2:
            segment_num = parts[-1]
    except:
        pass
    
    return {
        "type": segment_type,
        "module": module_num,
        "segment": segment_num
    }

# Function to format segment ID into a proper citation
def format_citation(segment_id, segment_title, segment_type="Live Session", module_videos=None):
    parts = segment_id.split('_')
    segment_info = extract_segment_info(segment_id)
    
    if parts[0] == "LS":
        if len(parts) >= 2:
            try:
                session_num = int(parts[1])
                return f"Live Session {session_num}, discussing {segment_title}, Foundations Course, 2025"
            except ValueError:
                return f"Live Session {parts[1]}, discussing {segment_title}, Foundations Course, 2025"
    elif parts[0] == "LS_Pro":
        return f"Live Session Pro, discussing {segment_title}, Foundations Course, 2025"
    elif parts[0] == "PER":
        if len(parts) >= 3:
            try:
                if "Module" in parts[1]:
                    module_num = int(parts[1].replace("Module", ""))
                else:
                    module_num = int(parts[1])
                
                # Get the official module title
                track_type = "Personal Track"
                module_title = get_official_module_title(track_type, module_num, segment_title)
                
                return f"{track_type}, Module {module_num}: {module_title}, Foundations Course, 2025"
            except (ValueError, IndexError):
                return f"Personal Track, {parts[1]}, {segment_title}, Foundations Course, 2025"
    elif parts[0] == "PRO":
        if len(parts) >= 3:
            try:
                if "Module" in parts[1]:
                    module_num = int(parts[1].replace("Module", ""))
                else:
                    module_num = int(parts[1])
                
                # Get the official module title
                track_type = "Professional Track"
                module_title = get_official_module_title(track_type, module_num, segment_title)
                
                return f"{track_type}, Module {module_num}: {module_title}, Foundations Course, 2025"
            except (ValueError, IndexError):
                return f"Professional Track, {parts[1]}, discussing {segment_title}, Foundations Course, 2025"
    elif parts[0] == "VC":
        # Handle Voice Clips
        voice_clip_titles = [
            "3rd Way", "Attachment and Core Love", "Contact", "Desire",
            "Emptiness and Existence", "Establishing Existence", "False Belonging",
            "Fragility, Loyalty, and Systems of Oppression", "Having", "Leadership",
            "Love", "Loyalty", "Narcissistic Body", "Now Time", "Order and Existence",
            "Restoring Order of Emptiness", "Unloved, Unlived"
        ]
        
        # Try to find best matching voice clip title
        best_match = None
        best_score = 0
        
        for vc_title in voice_clip_titles:
            vc_lower = vc_title.lower()
            segment_lower = segment_title.lower()
            
            # Check for word matches
            common_words = set(vc_lower.split()).intersection(set(segment_lower.split()))
            
            score = 0
            if common_words:
                score = len(common_words) / len(vc_lower.split())
                
            if score > best_score:
                best_score = score
                best_match = vc_title
        
        if best_match and best_score > 0.2:
            return f"Voice Clip - {best_match}, Foundations Course, 2025"
        
        return f"Voice Clip - {segment_title}, Foundations Course, 2025"
    
    return f"{segment_type} {segment_id}, Foundations Course, 2025"

def get_official_module_title(track_type, module_num, segment_title=None):
    """Return the official module title based on track type and module number."""
    
    # Convert module_num to integer
    try:
        module_num = int(module_num)
    except (ValueError, TypeError):
        module_num = 1  # Default to module 1
    
    # Define official module titles
    module_titles = {
        "Personal Track": {
            1: ["We Always Start with Existence", "Primary Existence and Relational Existence", "Qualification of Existence"],
            2: ["Surfing the Movement", "The Two Bodies: Adaptation and Emergence", "The Currency of the Adaptive Body"],
            3: ["Holding the Tension: Beyond Collapse and Control", "Breaking Reality Fields", "Mapping the Energetics of a Doublebind", 
                "Right Relationship with Power", "Map of the Primal Body and Surrounding Fields"],
            4: ["Addictions Are A Resistance to Evolution", "The Nature of Love"],
            5: ["Unbinding the Control Matrix", "Remembering the Receiving of Love"],
            6: ["Deepening Concentration Without Force", "You Can't Fuck Up Meditation", "The Practice of Taking Everything Back to Nothing"],
            7: ["The Existence of True Intimacy", "There is No Shortcut to Maturity", "Integration Between All Realities"],
            8: ["The Place Where Intimacy is Happening is Now", "Creating Velocity", "Flickering Between States of Consciousness",
                "Remediating Chronic Inflammation", "Applying the Physics within Relationality", "Instructions for Integration"]
        },
        "Professional Track": {
            1: ["Origins of the Model", "Being in the Liminal"],
            2: ["Importance of View", "Physics and Structure of Energetic Patterns"],
            3: ["Expanded Description of the Primal Body", "Central Pulse of Creation", 
                "Addiction and Abuse as Constellations of Consciousness"],
            4: ["The Self, Authority, and Conscientiousness"],
            5: ["Understanding Addiction and Exploitative Systems", "Fuck Around and Find Out"],
            6: ["Being with the Discomfort of Not Knowing", "Belonging and Independence"],
            7: ["The Erotic, Innocence, and Holding a Field of Love"],
            8: ["Solving the Algorithm for Emergence"]
        }
    }
    
    # Default if track or module not found
    if track_type not in module_titles or module_num not in module_titles[track_type]:
        return f"Module {module_num} Content"
    
    # Just return the first title for this module - this is safer and prevents hallucination
    return module_titles[track_type][module_num][0]

# Load the module video information
@st.cache_resource(show_spinner=False)
def load_module_videos(file_path="module_videos.json"):
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            # Default mapping if file doesn't exist
            return {
                "Personal Track Module 1": {
                    "thumbnail": "https://i.imgur.com/shJBySD.png",
                    "purchase_url": "https://the-center.circle.so/c/foundations-self-study",
                    "title": "We Always Start with Existence (3 mins)"
                }
            }
    except Exception as e:
        st.error(f"Error loading module videos: {e}")
        return {}

# Load the FAISS index, vectorizer, and segment metadata
@st.cache_resource(show_spinner=False)
def load_faiss_resources(vectordb_dir="vectordb"):
    index_path = os.path.join(vectordb_dir, "course_segments.index")
    metadata_path = os.path.join(vectordb_dir, "segments_metadata.json")
    vectorizer_path = os.path.join(vectordb_dir, "vectorizer.pkl")
    
    if not os.path.exists(index_path) or not os.path.exists(metadata_path) or not os.path.exists(vectorizer_path):
        st.error(f"Vector database files not found in {vectordb_dir}")
        return None, None, None
    
    try:
        index = faiss.read_index(index_path)
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            segments_metadata = json.load(f)
        
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        
        return index, segments_metadata, vectorizer
    except Exception as e:
        st.error(f"Error loading vector database files: {e}")
        return None, None, None

# Function to search for relevant segments
def search_segments(query, index, segments_metadata, vectorizer, limit=5):
    if index is None or segments_metadata is None or vectorizer is None:
        st.error("Vector database resources not loaded properly")
        return []
    
    try:
        clean_query = clean_text(query)
        query_vector = vectorizer.transform([clean_query]).toarray().astype('float32')
        distances, indices = index.search(query_vector, limit)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(segments_metadata):
                segment = segments_metadata[idx]
                segment_content = correct_names(segment["content"])
                
                results.append({
                    "content": segment_content,
                    "session_title": segment["session_title"],
                    "title": segment["title"],
                    "session_type": segment.get("session_type", "Unknown Type"),
                    "session_number": segment.get("session_number", 0),
                    "module_number": segment.get("module_number", 0),
                    "segment_number": segment["segment_number"],
                    "segment_id": segment["segment_id"],
                    "similarity_score": float(1.0 / (1.0 + distances[0][i]))
                })
        
        return results
    except Exception as e:
        st.error(f"Error during search: {e}")
        return []

def get_most_relevant_module(relevant_segments, query="", module_videos=None, response_text=None):
    """
    Select the most relevant module by analyzing citations in the response,
    and use the exact title from the citation.
    """
    if module_videos is None or not module_videos:
        return None
    
    # Map of all possible titles for each module
    module_titles = {
        1: [
            "We Always Start with Existence",
            "Primary Existence and Relational Existence",
            "Qualification of Existence"
        ],
        2: [
            "Surfing the Movement",
            "The Two Bodies: Adaptation and Emergence",
            "The Currency of the Adaptive Body"
        ],
        3: [
            "Holding the Tension: Beyond Collapse and Control",
            "Breaking Reality Fields",
            "Mapping the Energetics of a Doublebind",
            "Right Relationship with Power",
            "Map of the Primal Body and Surrounding Fields"
        ],
        4: [
            "Addictions Are A Resistance to Evolution",
            "The Nature of Love"
        ],
        5: [
            "Unbinding the Control Matrix",
            "Remembering the Receiving of Love"
        ],
        6: [
            "Deepening Concentration Without Force",
            "You Can't Fuck Up Meditation",
            "The Practice of Taking Everything Back to Nothing"
        ],
        7: [
            "The Existence of True Intimacy",
            "There is No Shortcut to Maturity",
            "Integration Between All Realities"
        ],
        8: [
            "The Place Where Intimacy is Happening is Now",
            "Creating Velocity",
            "Flickering Between States of Consciousness",
            "Remediating Chronic Inflammation",
            "Applying the Physics within Relationality",
            "Instructions for Integration"
        ]
    }
    
    # For text-based approach (using the actual response text)
    if response_text:
        # Pattern to extract full personal track citations
        citation_pattern = r'(\d+\.\s+Personal Track,\s+Module\s+(\d+):\s+(.*?),\s+Foundations Course, 2025)'
        personal_citations = re.findall(citation_pattern, response_text)
        
        if personal_citations:
            # Count modules and collect titles
            module_counts = {}
            module_titles_found = {}
            
            for full_citation, module_num, title in personal_citations:
                try:
                    module_num = int(module_num)
                    module_key = f"Personal Track Module {module_num}"
                    
                    if module_key not in module_counts:
                        module_counts[module_key] = 0
                        module_titles_found[module_key] = []
                    
                    module_counts[module_key] += 1
                    module_titles_found[module_key].append(title.strip())
                except ValueError:
                    continue
            
            # Select most frequently cited module
            if module_counts:
                best_module_key = max(module_counts, key=module_counts.get)
                module_num = int(best_module_key.split(' ')[-1])
                
                # Get the most frequent title for this module
                title_counts = {}
                for title in module_titles_found[best_module_key]:
                    if title not in title_counts:
                        title_counts[title] = 0
                    title_counts[title] += 1
                
                # Use the most frequently cited title
                best_title = max(title_counts, key=title_counts.get) if title_counts else module_titles[module_num][0]
                
                if best_module_key in module_videos:
                    # Create module info with original URL and thumbnail, but updated title
                    return {
                        "type": "Personal Track",
                        "module": str(module_num),
                        "video_thumbnail": module_videos[best_module_key]["thumbnail"],
                        "purchase_url": module_videos[best_module_key]["purchase_url"],
                        "video_title": best_title
                    }
    
    # Fallback to segment-based approach if no citations found
    filtered_segments = [s for s in relevant_segments if s.get('segment_id', '').startswith('PER_')]
    
    if not filtered_segments:
        default_key = "Personal Track Module 1"
        if default_key in module_videos:
            return {
                "type": "Personal Track",
                "module": "1",
                "video_thumbnail": module_videos[default_key]["thumbnail"],
                "purchase_url": module_videos[default_key]["purchase_url"],
                "video_title": module_videos[default_key]["title"]
            }
        return None
    
    # Standard segment-based approach (keeping original logic)
    module_counts = {}
    for segment in filtered_segments:
        segment_id = segment.get('segment_id', '')
        parts = segment_id.split('_')
        if len(parts) >= 2 and parts[0] == 'PER':
            try:
                module_num = parts[1]
                if module_num.startswith('Module'):
                    module_num = module_num.replace('Module', '')
                
                module_key = f"Personal Track Module {module_num}"
                
                if module_key not in module_counts:
                    module_counts[module_key] = 0
                module_counts[module_key] += 1
            except:
                continue
    
    if not module_counts:
        default_key = "Personal Track Module 1"
        if default_key in module_videos:
            return {
                "type": "Personal Track",
                "module": "1",
                "video_thumbnail": module_videos[default_key]["thumbnail"],
                "purchase_url": module_videos[default_key]["purchase_url"],
                "video_title": module_videos[default_key]["title"]
            }
        return None
    
    best_module_key = max(module_counts, key=module_counts.get)
    
    if best_module_key not in module_videos:
        default_key = list(module_videos.keys())[0]
        return {
            "type": "Personal Track",
            "module": default_key.split(' ')[-1],
            "video_thumbnail": module_videos[default_key]["thumbnail"],
            "purchase_url": module_videos[default_key]["purchase_url"],
            "video_title": module_videos[default_key]["title"]
        }
    
    return {
        "type": "Personal Track",
        "module": best_module_key.split(' ')[-1],
        "video_thumbnail": module_videos[best_module_key]["thumbnail"],
        "purchase_url": module_videos[best_module_key]["purchase_url"],
        "video_title": module_videos[best_module_key]["title"]
    }

# Function to generate a response with Claude
def generate_response(query, relevant_segments, conversation_history=None, module_videos=None):
    if anthropic_client is None:
        st.error("Claude API connection not available")
        return "Sorry, we cannot generate a response at this time due to API connection issues."
    
    if not relevant_segments:
        prompt = f"""
        The user is asking about the Movement of Existence (MoE) Foundations course. However, I couldn't find any relevant 
        segments in the transcripts. Please respond politely that you don't have specific information 
        on this topic from the course materials.
        
        User question: {query}
        """
        
        system_message = """Within the present role of extremely skilled facilitator and certified teacher of the Foundations course, please draw primarily on the course segments to provide a conversational answer to the query that is privy to the full complexity of connections that the asker is making (with the extreme skill of a lacanian analyst with decades of clinical experience and training in coherence therapy), and that is offering an attuned synthesis of relevent course segments to respond to the heart of the inquiry. In dialogue, you use a transference-focused psychotherapy (TFP) lens to inform your dynamically updated object-relations based treatment model, staying flexible to when intuition and art are required, and guide the user towards transformative improvement. Do not explain theory unless necessary, assume the client is well-versed and will ask questions if they don't understand. Don't mention names of techniques if unnecessary. Do not use lists. Speak conversationally. Do not speak like a blog post or wikipedia entry. Be economical in your speech. Center your sensemaking around concepts from the segments, and cite these appropriately.

Never use "I" when referring to yourself. Instead, use "we" when necessary.

Keep your response concise and focused - maximum 3 paragraphs. Be efficient with words while fully addressing the question.

Include appropriate academic citations for parts of your answer that draw on the course transcript segments. Use superscript numbers (e.g., This concept¹) and include a numbered reference list at the end of your response."""
        
        try:
            completion = anthropic_client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=800,
                temperature=0.3,
                system=system_message,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return completion.content[0].text
        except Exception as e:
            try:
                completion = anthropic_client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=800,
                    temperature=0.3,
                    system=system_message,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return completion.content[0].text
            except Exception as fallback_error:
                st.error(f"Error generating response with fallback model: {fallback_error}")
                return f"We encountered an error while generating a response. Please try again later."
    
    context = ""
    citation_references = {}
    citation_count = 1
    
    for i, segment in enumerate(relevant_segments):
        segment_id = segment.get('segment_id', f"segment_{i+1}")
        segment_title = segment.get('title', 'Untitled')
        citation_references[citation_count] = format_citation(segment_id, segment_title, segment.get('session_type', 'Unknown'), module_videos)
        
        context += f"--- Segment {i+1} (Citation {citation_count}) ---\n"
        context += f"Title: {segment_title}\n"
        context += f"Session: {segment['session_title']} (Segment {segment['segment_number']})\n"
        context += f"Segment ID: {segment_id}\n"
        context += f"Content: {segment['content']}\n\n"
        
        citation_count += 1
    
    citation_list = "\n".join([f"{i}. {ref}" for i, ref in citation_references.items()])
    
    conversation_context = ""
    if conversation_history and len(conversation_history) > 0:
        conversation_context = "Previous conversation:\n"
        for i, message in enumerate(conversation_history):
            if message["role"] == "user":
                conversation_context += f"User: {message['content']}\n"
            else:
                conversation_context += f"Assistant: {message['content']}\n"
        conversation_context += "\n"
    
    prompt = f"""
    You are answering a question about the Movement of Existence (MoE) Foundations course.
    
    {conversation_context}
    
    I'll provide you with relevant segments from the course transcripts. 
    Use this information to answer the question. While focusing primarily on these transcripts, 
    you can also draw upon your understanding of coherence therapy and transference-focused psychotherapy (TFP)
    when appropriate.
    
    When answering follow-up questions, don't repeat exact phrases or explanations from your previous responses.
    Express the same concepts in different words if needed.
    
    Here are the relevant segments from the course transcripts:
    
    {context}
    
    When you reference information from the segments in your answer, use superscript citation numbers 
    (like this¹) and include the following numbered references at the end of your response. Put each citation on its own line:
    
    {citation_list}
    
    Based primarily on these segments, please answer the following question:
    
    {query}
    """
    
    system_message = """Within the present role of extremely skilled facilitator and certified teacher of the Foundations course, please draw primarily on the course segments to provide a conversational answer to the query that is privy to the full complexity of connections that the asker is making (with the extreme skill of a lacanian analyst with decades of clinical experience and training in coherence therapy), and that is offering an attuned synthesis of relevent course segments to respond to the heart of the inquiry. In dialogue, you use a transference-focused psychotherapy (TFP) lens to inform your dynamically updated object-relations based treatment model, staying flexible to when intuition and art are required, and guide the user towards transformative improvement. Do not explain theory unless necessary, assume the client is well-versed and will ask questions if they don't understand. Don't mention names of techniques if unnecessary. Do not use lists. Speak conversationally. Do not speak like a blog post or wikipedia entry. Be economical in your speech. Center your sensemaking around concepts from the segments, and cite these appropriately.

Never use "I" when referring to yourself. Instead, use "we" when necessary.

Keep your response concise and focused - maximum 3 paragraphs. Be efficient with words while fully addressing the question.

Include appropriate academic citations for parts of your answer that draw on the course transcript segments. Use superscript numbers (e.g., This concept¹) and include a numbered reference list at the end of your response."""
    
    try:
        completion = anthropic_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=800,
            temperature=0.3,
            system=system_message,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return completion.content[0].text
    except Exception as e:
        try:
            completion = anthropic_client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=800,
                temperature=0.3,
                system=system_message,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return completion.content[0].text
        except Exception as fallback_error:
            st.error(f"Error generating response with fallback model: {fallback_error}")
            return f"We encountered an error while generating a response. Please try again later."

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []
    st.session_state.current_query = ""
    st.session_state.current_response = ""
    st.session_state.current_module = None
    st.session_state.in_chat = False

# Sticky header with title
st.markdown('<div class="app-header">', unsafe_allow_html=True)
st.markdown('<div class="app-title">Understanding Foundations</div>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Bree Greenberg, LMFT (2025)</p>', unsafe_allow_html=True)

# Only show New Chat button if a conversation is in progress
if st.session_state.in_chat:
    if st.button("New Chat", key="new_chat_btn", help="Start a new conversation"):
        st.session_state.conversation = []
        st.session_state.current_query = ""
        st.session_state.current_response = ""
        st.session_state.current_module = None
        st.session_state.in_chat = False
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Initialize variables to avoid "not defined" errors
index, segments_metadata, vectorizer, module_videos = None, None, None, {}

# Load resources
try:
    if "resources_loaded" not in st.session_state:
        index, segments_metadata, vectorizer = load_faiss_resources()
        module_videos = load_module_videos()
        
        # Only save to session state if successfully loaded
        if index is not None and segments_metadata is not None and vectorizer is not None:
            st.session_state.index = index
            st.session_state.segments_metadata = segments_metadata
            st.session_state.vectorizer = vectorizer
            st.session_state.module_videos = module_videos
            st.session_state.resources_loaded = True
    else:
        # Retrieve from session state
        index = st.session_state.index
        segments_metadata = st.session_state.segments_metadata
        vectorizer = st.session_state.vectorizer
        index = st.session_state.index
        segments_metadata = st.session_state.segments_metadata
        vectorizer = st.session_state.vectorizer
        module_videos = st.session_state.module_videos
except Exception as e:
    st.error(f"Error loading resources: {e}")

# Display current question and response
if st.session_state.current_query and st.session_state.current_response:
    st.markdown(f'<div class="message-container"><div class="user-message">{st.session_state.current_query}</div></div>', unsafe_allow_html=True)
    
    # Format citations in the response
    formatted_response = format_citations_html(st.session_state.current_response)
    
    # Convert newlines to proper paragraph tags for better spacing
    formatted_response = formatted_response.replace("\n\n", "</p><p>")
    formatted_response = f"<p>{formatted_response}</p>"
    
    st.markdown(f'<div class="message-container"><div class="assistant-message">{formatted_response}</div></div>', unsafe_allow_html=True)
    
    # Display module highlight only if there's a current response
    if st.session_state.current_module:
        # Get video information from the module with fallbacks for all values
        module_type = st.session_state.current_module.get("type", "Personal Track")
        module_num = st.session_state.current_module.get("module", "1")
        module_title = st.session_state.current_module.get("video_title", "We Always Start with Existence (3 mins)")
        
        # Get purchase URL and thumbnail with fallbacks
        purchase_url = st.session_state.current_module.get("purchase_url", "https://the-center.circle.so/c/foundations-self-study")
        thumbnail = st.session_state.current_module.get("video_thumbnail", "https://i.imgur.com/shJBySD.png")
        
        # Build the module display with a clickable thumbnail
        st.markdown(f'''
        <div class="module-highlight">
            <div class="module-title">Listen to Bree: {module_title}</div>
            <a href="{purchase_url}" target="_blank">
                <img src="{thumbnail}" alt="Bree Greenberg - {module_type} {module_num}" style="max-width: 100%;">
            </a>
            <div style="text-align: center; margin-top: 0.5rem;">
                <a href="{purchase_url}" target="_blank" style="color: #317485; text-decoration: none;">
                    Access this module →
                </a>
            </div>
        </div>
        ''', unsafe_allow_html=True)

# Different UI depending on whether we're in a chat or not
if not st.session_state.in_chat:
    # Initial state with centered input
    st.markdown('<div class="empty-state-container">', unsafe_allow_html=True)
    st.markdown('''<p class="initial-prompt">Let's play. What's your question?</p>''', unsafe_allow_html=True)
    
    with st.form(key="query_form_initial", clear_on_submit=True):
        col1, col2 = st.columns([10, 1])
        with col1:
            query = st.text_input("Question", key="user_input_initial", label_visibility="collapsed")
        with col2:
            submit_button = st.form_submit_button(label="Ask")
    
    st.markdown('</div>', unsafe_allow_html=True)
else:
    # Follow-up state with simpler input
    st.markdown('<div class="followup-form">', unsafe_allow_html=True)
    with st.form(key="query_form_followup", clear_on_submit=True):
        query = st.text_input("Follow-up question", key="user_input_followup", label_visibility="collapsed")
        submit_button = st.form_submit_button(label="Continue Chat")
    st.markdown('</div>', unsafe_allow_html=True)

# Process the query when submitted
if submit_button and query:
    # Store the current query
    st.session_state.current_query = query
    
    # Set in_chat state to true
    st.session_state.in_chat = True
    
    # Add to conversation history
    st.session_state.conversation.append({"role": "user", "content": query})
    
    # Display loading state
    loading_placeholder = st.empty()
    loading_placeholder.markdown('<div class="loading-indicator">Checking archives...</div>', unsafe_allow_html=True)
    
    # Search for relevant segments
    relevant_segments = []
    if index is not None and segments_metadata is not None and vectorizer is not None:
        relevant_segments = search_segments(query, index, segments_metadata, vectorizer)
    
    # Get relevant conversation history
    conversation_history = st.session_state.conversation[:-1] if len(st.session_state.conversation) > 1 else None
    
    # Generate response
    response = generate_response(query, relevant_segments, conversation_history, module_videos)
    
    # Determine most relevant module - PASS THE RESPONSE TEXT HERE
    most_relevant_module = get_most_relevant_module(relevant_segments, query, module_videos, response)
    
    # Debug logging
    print(f"Selected module: {most_relevant_module}")
    
    # Store results in session state
    st.session_state.current_response = response
    st.session_state.current_module = most_relevant_module
    
    # Add assistant response to conversation history
    st.session_state.conversation.append({"role": "assistant", "content": response})
    
    # Clear loading state
    loading_placeholder.empty()
    
    # Rerun to update UI
    st.rerun()
