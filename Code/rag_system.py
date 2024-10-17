import os
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
import sys
import pickle
from sklearn.metrics.pairwise import cosine_similarity

class RAGSystem:
    def __init__(self, papers_folder=None, chunk_size=10000, chunk_overlap=800, index_file='faiss_index_100_papers.pickle'):
        self.index_file = index_file
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
       
        self.model = SentenceTransformer('all-MiniLM-L12-v2')

        if papers_folder:
            print(f"Assigning chunksize = {chunk_size}, chunkoverlap = {chunk_overlap}...")
            self.papers = self.read_papers(papers_folder)
            self.chunks, self.chunk_metadata = self.create_chunks()
            self.index = self.create_vector_db()
            self.save_index()
        else:
            self.load_index()

    def read_papers(self, folder_path):
        print(f"Reading papers in {folder_path} folder...")
        papers = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.pdf'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    content = ""
                    for page in pdf_reader.pages:
                        content += page.extract_text()
                    papers.append({
                        'title': filename,
                        'content': content
                    })
        return papers

    def create_chunks(self):
        print("Creating chunks...")
        chunks = []
        chunk_metadata = []
        for paper in self.papers:
            paper_chunks = [paper['content'][i:i+self.chunk_size] for i in range(0, len(paper['content']), self.chunk_size-self.chunk_overlap)]
            chunks.extend(paper_chunks)
            chunk_metadata.extend([{'title': paper['title'], 'chunk_index': i} for i in range(len(paper_chunks))])
        return chunks, chunk_metadata

    def create_vector_db(self):
        print("Creating Vector db...")
        embeddings = self.model.encode(self.chunks)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        return index

    def save_index(self):
        print(f"Saving index to {self.index_file}...")
        with open(self.index_file, 'wb') as f:
            pickle.dump((self.index, self.chunk_metadata, self.chunks), f)

    def load_index(self):
        print(f"Loading index from {self.index_file}...")
        with open(self.index_file, 'rb') as f:
            self.index, self.chunk_metadata, self.chunks = pickle.load(f)

    def get_relevant_chunks(self, query, top_k=5, similarity_threshold=0.5):
        print("Getting relevant chunks...")
        query_vector = self.model.encode([query]).astype('float32')
        
        
        k = min(top_k * 2, len(self.chunks))
        distances, indices = self.index.search(query_vector, k)
        
        
        candidate_embeddings = self.model.encode([self.chunks[i] for i in indices[0]])
        similarities = cosine_similarity(query_vector, candidate_embeddings)[0]
        
      
        sorted_indices = np.argsort(similarities)[::-1]
        filtered_indices = [indices[0][i] for i in sorted_indices if similarities[i] >= similarity_threshold][:top_k]
        
        relevant_chunks = [self.chunks[i] for i in filtered_indices]
        chunk_metadata = [self.chunk_metadata[i] for i in filtered_indices]
        
        return relevant_chunks, chunk_metadata

    def generate_response(self, query):
        print("Generating response...")
        relevant_chunks, chunk_metadata = self.get_relevant_chunks(query)

       
        if len(relevant_chunks) == 0:
            return "I'm sorry, but I couldn't find any relevant information to answer your query. Could you please provide a more specific or detailed question?"

        context = "\n".join([f"From paper '{meta['title']}', chunk {meta['chunk_index']}:\n{chunk}" 
                             for chunk, meta in zip(relevant_chunks, chunk_metadata)])

        prompt = f"Based on the following excerpts from research papers, answer the question: {query}\n\nContext:\n{context} in 100  words. If the question cannot be answered based solely on the given context, state that there is insufficient information to provide a complete answer."
       
        genai.configure(api_key='AIzaSyAR8TLAcQSgj07VUo2-A4_CUY0WeoseRdw')
        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        response = model.generate_content(prompt)

       
        if "insufficient information" in response.text.lower():
            return response.text  # Return without references

        
        cited_papers = set(meta['title'] for meta in chunk_metadata)
        if len(cited_papers) > 0:
            citations = "References:\n" + "\n".join(cited_papers)
            return response.text + "\n\n" + citations
        else:
            return response.text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rag_system.py <papers_folder>")
        sys.exit(1)
    
    papers_folder = sys.argv[1]
    rag_system = RAGSystem(papers_folder)
    print(f"RAG system initialized with papers from {papers_folder}")
    print(f"Loaded {len(rag_system.papers)} papers")
    print(f"Created {len(rag_system.chunks)} chunks for vector search")
