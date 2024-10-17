# ASAPP_hackathon

Steps to run the code.

## Table of Content
 - [Repository Structure](#repository-structure)
 - [Requirements](#requirements)
 - [To run the System](#to-run-the-system)

## Repository Structure
ASAPP_Hackathon/<br>
Code/<br>
&nbsp;&nbsp;&nbsp;&nbsp;├── app.py<br>
&nbsp;&nbsp;&nbsp;&nbsp;├── faiss_index_100_papers.pickle<br>
&nbsp;&nbsp;&nbsp;&nbsp;├── papers/<br>
&nbsp;&nbsp;&nbsp;&nbsp;├── rag_system.py<br>
Team_Decoders.pptx<br>
READMe.md<br>
requirements.txt

## Requirements
- Download and Install Python from offcial website.<br>
- To install required libraries run command : <br>
```bash
  pip install -r requirements.txt
```
- Upload the research papers in "Code/papers" folder

## To run the system
- The "faiss_index_100_papers.pickle" contains the index (knowlege base) generated using FAISS (Facebook AI Similarity Search) and  pickle module generated on our system.
- To generate the pickle module run command : <br>
```bash
python .\rag_system papers
```
- Here "papers" is the folder which contains reasearch papers.
- Now , to query run command :<br>
```bash
streamlit run app.py
```
- A input text box will be there enter your query and click "Enter" and the required answer will be displayed with relevant research papers.
  







