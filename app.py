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
        font-family: 'Futura', 'Trebuchet MS', sans-serif;
        color: white;
        font-size: 1.5rem;
        font-weight: normal;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        font-family: 'Futura', 'Trebuchet MS', sans-serif;
        color: #ccc;
        font-size: 0.9rem;
        margin-top: 0;
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
    
    /* Button hover styles - override Streamlit's default red hover */
    .stButton > button:hover,
    .stForm button:hover,
    button[kind="formSubmit"]:hover,
    button[kind="primary"]:hover,
    [data-testid="stFormSubmitButton"] button:hover {
        background-color: #3b8a9d !important; /* Lighter shade of the accent blue */
        border-color: #3b8a9d !important;
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
    .loading-indicator {
        color: #317485;
        font-style: italic;
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
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Initialize Anthropic client
try:
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
def format_citation(segment_id, segment_title, segment_type="Live Session"):
    parts = segment_id.split('_')
    segment_info = extract_segment_info(segment_id)
    
    if parts[0] == "LS":
        if len(parts) >= 2:
            try:
                session_num = int(parts[1])
                segment_detail = f"Segment {parts[-1]}" if len(parts) > 2 else ""
                return f"Live Session {session_num}, {segment_detail}, Foundations Course, 2025"
            except ValueError:
                segment_detail = f"Segment {parts[-1]}" if len(parts) > 2 else ""
                return f"Live Session {parts[1]}, {segment_detail}, Foundations Course, 2025"
    elif parts[0] == "LS_Pro":
        segment_detail = f"Segment {parts[-1]}" if len(parts) > 1 else ""
        return f"Live Session Pro, {segment_detail}, Foundations Course, 2025"
    elif parts[0] == "PER":
        if len(parts) >= 3:
            try:
                if "Module" in parts[1]:
                    module_num = int(parts[1].replace("Module", ""))
                else:
                    module_num = int(parts[1])
                segment_detail = f"Segment {parts[-1]}" if len(parts) > 3 else ""
                return f"Personal Track - Module {module_num}, {segment_title}, {segment_detail}, Foundations Course, 2025"
            except (ValueError, IndexError):
                segment_detail = f"Segment {parts[-1]}" if len(parts) > 3 else ""
                return f"Personal Track - {parts[1]}, {segment_title}, {segment_detail}, Foundations Course, 2025"
    elif parts[0] == "PRO":
        if len(parts) >= 3:
            try:
                if "Module" in parts[1]:
                    module_num = int(parts[1].replace("Module", ""))
                else:
                    module_num = int(parts[1])
                segment_detail = f"Segment {parts[-1]}" if len(parts) > 3 else ""
                return f"Professional Track - Module {module_num}, {segment_title}, {segment_detail}, Foundations Course, 2025"
            except (ValueError, IndexError):
                segment_detail = f"Segment {parts[-1]}" if len(parts) > 3 else ""
                return f"Professional Track - {parts[1]}, {segment_title}, {segment_detail}, Foundations Course, 2025"
    elif parts[0] == "VC":
        segment_detail = f"Segment {parts[-1]}" if len(parts) > 2 else ""
        return f"Voice Clip - {segment_title}, {segment_detail}, Foundations Course, 2025"
    
    return f"{segment_type} {segment_id}, Foundations Course, 2025"

# Function to search for relevant segments
def search_segments(query, index, segments_metadata, vectorizer, limit=5):
    if not relevant_segments or len(relevant_segments) == 0:
        return None
    
    module_scores = {}
    
    for segment in relevant_segments:
        segment_id = segment.get('segment_id', '')
        segment_info = extract_segment_info(segment_id)
        
        if not segment_info or not segment_info["type"]:
            continue
        
        module_key = f"{segment_info['type']}"
        if segment_info["module"]:
            module_key += f" {segment_info['module']}"
        
        segment_title = segment.get('title', 'Untitled')
        score = segment.get('similarity_score', 0)
        
        if module_key not in module_scores:
            module_scores[module_key] = {
                "total_score": 0,
                "count": 0,
                "title": segment_title,
                "type": segment_info["type"],
                "module": segment_info["module"],
                "segment_id": segment_id
            }
        
        module_scores[module_key]["total_score"] += score
        module_scores[module_key]["count"] += 1
    
    best_module = None
    best_score = 0
    
    for module_key, data in module_scores.items():
        avg_score = data["total_score"] / data["count"] if data["count"] > 0 else 0
        if avg_score > best_score:
            best_score = avg_score
            best_module = data
    
    return best_module

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
                "Live Session 1": {
                    "thumbnail": "https://i.imgur.com/yTqsldK.png",
                    "purchase_url": "https://the-center.circle.so/c/foundations-self-study",
                    "title": "Foundations Course"
                }
            }
    except Exception as e:
        st.error(f"Error loading module videos: {e}")
        return {}
    index_path = os.path.join(vector_db_dir, "course_segments.index")
    metadata_path = os.path.join(vector_db_dir, "segments_metadata.json")
    vectorizer_path = os.path.join(vector_db_dir, "vectorizer.pkl")
    
    if not os.path.exists(index_path) or not os.path.exists(metadata_path) or not os.path.exists(vectorizer_path):
        st.error(f"Vector database files not found in {vector_db_dir}")
        return None, None, None
    
    index = faiss.read_index(index_path)
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        segments_metadata = json.load(f)
    
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    
    return index, segments_metadata, vectorizer

# Load the FAISS index, vectorizer, and segment metadata
@st.cache_resource(show_spinner=False)
def load_faiss_resources(vector_db_dir="vectordb"):
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

# Function to generate a response with Claude
def generate_response(query, relevant_segments, conversation_history=None):
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
        citation_references[citation_count] = format_citation(segment_id, segment_title, segment.get('session_type', 'Unknown'))
        
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
st.markdown('<h1 class="app-title">Understanding Foundations</h1>', unsafe_allow_html=True)
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

# Load resources
try:
    if "resources_loaded" not in st.session_state:
        index, segments_metadata, vectorizer = load_faiss_resources()
        module_videos = load_module_videos()
        st.session_state.index = index
        st.session_state.segments_metadata = segments_metadata
        st.session_state.vectorizer = vectorizer
        st.session_state.module_videos = module_videos
        st.session_state.resources_loaded = True
    else:
        index = st.session_state.index
        segments_metadata = st.session_state.segments_metadata
        vectorizer = st.session_state.vectorizer
        module_videos = st.session_state.module_videos
except Exception as e:
    st.error(f"Error loading resources: {e}")
    index, segments_metadata, vectorizer, module_videos = None, None, None, {}

# Display current question and response
if st.session_state.current_query and st.session_state.current_response:
    st.markdown(f'<div class="message-container"><div class="user-message">{st.session_state.current_query}</div></div>', unsafe_allow_html=True)
    
    # Format citations in the response
    formatted_response = format_citations_html(st.session_state.current_response)
    
    # Convert newlines to proper paragraph tags for better spacing
    formatted_response = formatted_response.replace("\n\n", "</p><p>")
    formatted_response = f"<p>{formatted_response}</p>"
    
    st.markdown(f'<div class="message-container"><div class="assistant-message">{formatted_response}</div></div>', unsafe_allow_html=True)
    
    # Display module highlight if available
    if st.session_state.current_module:
        module_container = st.container()
        with module_container:
            # Get video information from the module
            module_type = st.session_state.current_module.get("type", "")
            module_num = st.session_state.current_module.get("module", "")
            module_title = st.session_state.current_module.get("video_title", "Foundations Course")
            
            # Get purchase URL and thumbnail with fallbacks
            purchase_url = st.session_state.current_module.get("purchase_url", "https://the-center.circle.so/c/foundations-self-study")
            thumbnail = st.session_state.current_module.get("video_thumbnail", "https://i.imgur.com/yTqsldK.png")
            
            # Build the module display with a clickable thumbnail
            module_container.markdown(f'''
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
    st.markdown('<p class="initial-prompt">What would you like to know?</p>', unsafe_allow_html=True)
    
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
    loading_placeholder.markdown('<div class="loading-indicator">Thinking...</div>', unsafe_allow_html=True)
    
    # Search for relevant segments
    relevant_segments = []
    if index is not None and segments_metadata is not None and vectorizer is not None:
        relevant_segments = search_segments(query, index, segments_metadata, vectorizer)
    
    # Get relevant conversation history
    conversation_history = st.session_state.conversation[:-1] if len(st.session_state.conversation) > 1 else None
    
    # Generate response
    response = generate_response(query, relevant_segments, conversation_history)
    
    # Determine most relevant module
    most_relevant_module = get_most_relevant_module(relevant_segments, module_videos)
    
    # Store results in session state
    st.session_state.current_response = response
    st.session_state.current_module = most_relevant_module
    
    # Add assistant response to conversation history
    st.session_state.conversation.append({"role": "assistant", "content": response})
    
    # Clear loading state
    loading_placeholder.empty()
    
    # Rerun to update UI
    st.rerun()
