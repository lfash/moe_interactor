import os
import json
import sys
import re
import anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
print(f"Loaded API key: {'Key found' if ANTHROPIC_API_KEY else 'Key missing!'}")

def segment_transcript(transcript_file, output_dir, max_segment_size=1500, overlap=300):
    """
    Comprehensively segment a transcript into meaningful chunks
    
    Args:
    - transcript_file: Path to the transcript file
    - output_dir: Directory to save segment files
    - max_segment_size: Maximum characters per segment
    - overlap: Number of characters to overlap between segments
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    # Read transcript with proper encoding
    try:
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcript_text = f.read()
    except UnicodeDecodeError:
        with open(transcript_file, 'r', encoding='latin-1') as f:
            transcript_text = f.read()

    # Clean text of non-ASCII characters
    transcript_text = re.sub(r'[^\x00-\x7F]+', ' ', transcript_text)
    
    # Get episode name
    episode_name = os.path.basename(transcript_file).replace('.txt', '')
    
    print(f"Segmenting {episode_name}...")
    
    # Function to generate segment titles
    def generate_segment_title(segment_text):
        """Use Claude to generate a brief title for a segment"""
        try:
            response = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=50,
                temperature=0,
                messages=[
                    {"role": "user", "content": f"""Please provide a very brief (3-5 word) title that captures the main topic of this text segment:

{segment_text[:1000]}

Respond ONLY with the title."""}
                ]
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"Error generating title: {e}")
            return f"Segment of {episode_name}"

    # Comprehensive segmentation
    segments_created = 0
    for start in range(0, len(transcript_text), max_segment_size - overlap):
        # Extract segment
        end = min(start + max_segment_size, len(transcript_text))
        segment_text = transcript_text[start:end]
        
        # Generate title for the segment
        segment_title = generate_segment_title(segment_text)
        
        # Create segment file
        segment_filename = f"{episode_name}_segment_{segments_created+1:03d}.txt"
        segment_path = os.path.join(output_dir, segment_filename)
        
        with open(segment_path, 'w', encoding='utf-8') as f:
            f.write(f"Title: {segment_title}\n")
            f.write(f"Episode: {episode_name}\n")
            f.write(f"Segment: {segments_created+1}\n\n")
            f.write(segment_text)
        
        segments_created += 1
        
        # Stop if we've reached the end of the transcript
        if end == len(transcript_text):
            break
    
    print(f"Created {segments_created} segments for {episode_name}")
    return segments_created

def process_all_transcripts(transcripts_dir, segments_dir):
    """
    Process all transcripts in a directory
    """
    if not os.path.exists(segments_dir):
        os.makedirs(segments_dir)
    
    files_processed = 0
    total_segments = 0
    
    for filename in os.listdir(transcripts_dir):
        if filename.endswith(".txt"):
            transcript_path = os.path.join(transcripts_dir, filename)
            segments = segment_transcript(transcript_path, segments_dir)
            if segments:
                total_segments += segments
                files_processed += 1
    
    print(f"Processed {files_processed} transcript files")
    print(f"Created {total_segments} total segments")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python segment.py transcripts_directory segments_directory")
    else:
        transcripts_dir = sys.argv[1]
        segments_dir = sys.argv[2]
        process_all_transcripts(transcripts_dir, segments_dir)