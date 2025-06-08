from flask import Flask, request, jsonify
import os
import nltk
from sentence_transformers import SentenceTransformer, util
from langchain_community.document_loaders import PyPDFLoader
from transformers import AutoTokenizer, T5ForConditionalGeneration, pipeline, AutoModelForSeq2SeqLM
import torch
from langdetect import detect

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load English and French models
english_checkpoint = "LaMini-Flan-T5-248M"
english_tokenizer = AutoTokenizer.from_pretrained(english_checkpoint, legacy=False)
english_model = T5ForConditionalGeneration.from_pretrained(english_checkpoint).to(device)

french_checkpoint = "mbart-mlsum-automatic-summarization"
french_tokenizer = AutoTokenizer.from_pretrained(french_checkpoint, src_lang="fr_XX", legacy=False)
french_model = AutoModelForSeq2SeqLM.from_pretrained(french_checkpoint).to(device)

# Load the SentenceTransformer model for semantic chunking and self-similarity
semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

nltk.download('punkt')

def file_preprocessing(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    texts = [text.page_content for text in pages]

    # Tokenize text into sentences
    sentences = [sentence for text in texts for sentence in nltk.sent_tokenize(text)]
    
    # Create embeddings for the sentences
    embeddings = semantic_model.encode(sentences, convert_to_tensor=True)

    # Define chunking parameters
    max_chunk_size = 800
    chunks = []
    chunk = []
    chunk_size = 0

    for sentence, embedding in zip(sentences, embeddings):
        if chunk_size + len(sentence) > max_chunk_size:
            chunks.append(" ".join(chunk))
            chunk = []
            chunk_size = 0
        chunk.append(sentence)
        chunk_size += len(sentence)

    if chunk:
        chunks.append(" ".join(chunk))
    
    return chunks

def calculate_self_similarity(summary):
    sentences = nltk.sent_tokenize(summary)
    embeddings = semantic_model.encode(sentences, convert_to_tensor=True)
    similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)
    average_similarity = torch.mean(similarity_matrix).item()
    return average_similarity

@app.route('/summarize', methods=['POST'])
def summarize():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file provided"}), 400

    filepath = os.path.join("temp_files", file.filename)
    os.makedirs("temp_files", exist_ok=True)
    file.save(filepath)

    texts = file_preprocessing(filepath)

    # Detect the language of the first chunk of text
    detected_language = detect(texts[0])

    if detected_language == 'fr':
        tokenizer = french_tokenizer
        model = french_model
    else:
        tokenizer = english_tokenizer
        model = english_model

    # Initialize summarization pipeline with explicit device handling
    pipe_sum = pipeline(
        'summarization',
        model=model,
        tokenizer=tokenizer,
        device=device.index if torch.cuda.is_available() else -1
    )

    summaries = []
    for text in texts:
        input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
        input_length = len(input_ids[0])
        
        if input_length > 512:
            input_ids = input_ids[:, :512]
            input_length = 512

        chunk_summary_length = max(50, int(input_length * 0.1))
        max_length = min(chunk_summary_length, 512)
        min_length = max(30, int(chunk_summary_length * 0.5))

        result = pipe_sum(text, max_length=max_length, min_length=min_length)
        summaries.append(result[0]['summary_text'])

    final_summary = " ".join(summaries)

    # Calculate Self-Similarity (Coherence) score
    self_similarity_score = calculate_self_similarity(final_summary)
    print(self_similarity_score)
    return jsonify({
        "summary": final_summary,
    })

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
