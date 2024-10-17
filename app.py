from langchain.schema import Document
from langchain.chat_models import ChatOpenAI  # Fallback to OpenAI if needed
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI

# Set up Google API key (replace with your actual API key)
# Ensure this is set correctly
os.environ["GOOGLE_API_KEY"] = 'AIzaSyBcAnZl-Ts2bIwGuJFD6HCOilCuHEloWRc'

# Load the PDF and extract text
pdf_path = 'Methilesh_Resume.pdf'  # Replace with your PDF file path
pdfreader = PdfReader(pdf_path)

# Extract text from all pages
text = ''
for page in pdfreader.pages:
    content = page.extract_text()
    if content:
        text += content

# Split text into manageable chunks for summarization
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_text(text)

# Wrap the text chunks in Document objects
documents = [Document(page_content=t) for t in texts]

# Set up the LLM summarization chain using the Gemini model
# Ensure this class exists
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
summarize_chain = load_summarize_chain(llm, chain_type="map_reduce")

# Generate the summary
summary = summarize_chain.run(documents)
print("Summary of PDF:", summary)