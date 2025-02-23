# Created: 02-2025     By: shizuniik
# ### Text Summarizer: simple text summarizer using Hugging Face transformers.

# It shows only error messages from the transformers library.
from transformers.utils import logging
logging.set_verbosity_error()

from transformers import pipeline
import torch
import os
import gc

# --- Function Definitions ---
def load_text(txt_file: str) -> str:
    """
    Reads and returns the content of the given text file.
    
    :param txt_file: Path to the text file.
    :return: Content of the file.
    :raises FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"The file {txt_file} does not exist.")
    
    with open(txt_file, "r") as f:
        article = f.read()
    return article

def summarize(article: str, max_length: int, min_length: int) -> list:
    """
    Summarizes the input article using the pre-loaded summarization model.
    
    :param article: Text to be summarized.
    :param max_length: Maximum length of the summary.
    :param min_length: Minimum length of the summary.
    :return: Summary of the article.
    """
    return summarizer(article, 
                      max_length=max_length, 
                      min_length=min_length, 
                      do_sample=False)

def print_summary(summary: list) -> None:
    """
    Prints the summarized text.
    
    :param summary: Summary list returned by the summarizer.
    """
    print("Summary:", summary[0]['summary_text'])

def cleanup_memory():
    """
    Frees up memory used by the summarizer model.
    """
    global summarizer
    del summarizer
    gc.collect()

# --- Main Execution ---
if __name__ == "__main__":
    print("Loading summarization model...")
    # Load the summarization model
    summarizer = pipeline("summarization", 
                          model="facebook/bart-large-cnn",
                          torch_dtype=torch.bfloat16)
    print("Model loaded successfully.")

    # Example text for summarization
    try:
        print("Loading article text...")
        article = load_text("article.txt")   
        print("Article loaded:\n", article, "\n")
        
        print("Summarizing article...")
        summary = summarize(article, 130, 30)
        print("Summary generated.")
        
        print_summary(summary)
    except FileNotFoundError as e:
        print(e)

    # Free up memory used by the summarizer model
    cleanup_memory()