import os
import re
import logging
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv(override=True)

# Fetch API Key securely
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("❌ GROQ_API_KEY is missing. Please check your .env file!")

# Initialize Groq client with API Key
groq = Groq(api_key=api_key)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def classify_with_llm(log_msg):
    """
    Classifies a log message into one of the predefined categories using Groq's LLM.

    Categories:
    - Workflow Error
    - Deprecation Warning
    - Unclassified (fallback if no match found)

    Parameters:
    - log_msg (str): The log message to classify.

    Returns:
    - category (str): The predicted category.
    """

    prompt = f'''Classify the log message into one of these categories: 
    (1) Workflow Error, (2) Deprecation Warning.
    If you can't determine a category, return "Unclassified".
    Put the category inside <category> </category> tags. 
    Log message: {log_msg}'''

    try:
        # Send request to LLM
        chat_completion = groq.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="deepseek-r1-distill-llama-70b",  
            temperature=0.5
        )

        # Extract the category from the response
        content = chat_completion.choices[0].message.content
        match = re.search(r'<category>(.*?)<\/category>', content, flags=re.DOTALL)
        category = match.group(1) if match else "Unclassified"

        logger.info(f"Log: {log_msg} | Classified as: {category}")
        return category

    except Exception as e:
        logger.error(f"Error in classify_with_llm: {str(e)}")
        return "Unclassified"