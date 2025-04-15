from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import sent_tokenize
from torch.nn.functional import cosine_similarity
import numpy as np
import nltk
import os
import torch
import gc

class SemanticEmbedder:

    def __init__(
            self,
            semanticChecker:dict = {
                "folder": "BioBert/",
                "name": "dmis-lab/biobert-v1.1"
            },
            sentenceTokenizer:dict = {
                "folder": "Sentence-Tokenizers/",
                "name": "punkt_tab"
            }
        ):
        # Semantic Checker to load
        self.semanticChecker = {
            'path': f"Assets/Semantic-Checkers/{semanticChecker['folder']}",
            'name': semanticChecker['name']
        }
        self.sentenceTokenizer = {
            'path': f"Assets/{sentenceTokenizer['folder']}",
            'name': sentenceTokenizer['name']
        } 
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"        
        self.TOKENIZER:AutoTokenizer = None 
        self.MODEL:AutoModel = None
        self.LOADED = False
        self.document:str = None
        self.embeddings:np.ndarray = None
        self.sentences:np.ndarray = None
        self.includedChecks:np.ndarray = None # Same dimension as sentences
        
        # Download Model for Semantic Checking
        if not os.path.exists(self.semanticChecker['path']):
            os.makedirs(self.semanticChecker['path'])
            print(f"Downloading Semantic Checker: {self.semanticChecker['name']}")
            snapshot_download(repo_id=self.semanticChecker['name'], local_dir=self.semanticChecker['path'], local_dir_use_symlinks=False, repo_type="model")
            print("Download Complete\n")
        
        # Download Model for Sentence Segmentation
        if not os.path.exists(self.sentenceTokenizer['path']):
            os.makedirs(self.sentenceTokenizer['path'])
            print(f"Downloading Sentence Tokenizer: {self.sentenceTokenizer['name']}")
            # nltk.download(info_or_id=self.sentenceTokenizer['name'])
            nltk.download(self.sentenceTokenizer['name'], download_dir=self.sentenceTokenizer['path'])
            print("Download Complete\n")
        nltk.data.path.append(os.path.join(os.path.curdir, self.sentenceTokenizer['path']))

    def load(self):
        self.TOKENIZER = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=self.semanticChecker['path'],
        local_files_only=True
    )
        self.MODEL = AutoModel.from_pretrained(
            pretrained_model_name_or_path=self.semanticChecker['path'],
            local_files_only=True
        )
        self.MODEL.to(self.DEVICE)
        self.MODEL.eval()
        self.LOADED = True

    def unload(self):
        self.TOKENIZER = None
        self.MODEL = None
        gc.collect()
        if "cuda" in self.DEVICE:
            torch.cuda.empty_cache()
        self.LOADED = False

    def getEmbedding(self, sentence:str):
        wasLoaded = True
        if not self.LOADED: 
            self.load()
            wasLoaded = False
        inputs = self.TOKENIZER(sentence, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.MODEL(**inputs)
        
        # Attention Mask to ignore padding tokens
        attention_mask = inputs['attention_mask']

        # Mean Pooling
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        if not wasLoaded: self.unload()
        return (sum_embeddings / sum_mask)[0].cpu().numpy()
    
    def getSemanticSimilarity(self, sent1:str, sent2:str, show:bool = False):
        embedding1 = torch.tensor(self.getEmbedding(sent1)).unsqueeze(0)
        embedding2 = torch.tensor(self.getEmbedding(sent2)).unsqueeze(0)
        sim = cosine_similarity(embedding1, embedding2).item()
        if show:
            print(f"\n\t\tSEMANTIC SIMILIARITY = {sim: 0.4}")
            print(f"\t\t|---> Sentence 1: {sent1}")
            print(f"\t\t|---> Sentence 2: {sent2}")
        return sim

    def getOverlappingChunks(text, window_size=5, stride=3):
        # Sliding Window Chunks
        sentences = sent_tokenize(text)
        chunks = []

        for i in range(0, len(sentences), stride):
            chunk = ' '.join(sentences[i:i + window_size])
            if chunk:
                chunks.append(chunk)

        return chunks
    
    def initRAG(self, text:str):
        self.document = text
        self.sentences = np.array(sent_tokenize(text))
        # embeddings shape: (num_sentences, embedding_dim)
        self.embeddings = torch.stack([torch.tensor(self.getEmbedding(sent)) for sent in self.sentences]).numpy()
        self.includedChecks = np.zeros(len(self.sentences), dtype=bool)
        return

    def _chunkify(self, targetIdx:int, currentSim:float, minSim:float, minExpand:int=4, maxExpand=10):
        startIdx = max(0, targetIdx - maxExpand - 5)
        endIdx = targetIdx + maxExpand + 5
        sentences = self.sentences[startIdx:endIdx]

        # Adjust the target index relative to the sliced list
        adjustedTargetIdx = targetIdx - startIdx
        target = sentences[adjustedTargetIdx]

        bestScore = currentSim
        bestChunk = ""
        bestSpan = (None, None)
        n = len(sentences)
        
        for windowSize in range(minExpand, maxExpand + 1):
            for start in range(max(0, adjustedTargetIdx - windowSize + 1), min(adjustedTargetIdx + 1, n - windowSize + 1)):
                end = start + windowSize
                chunk = " ".join(sentences[start:end])
            
                if target not in chunk:
                    continue  # ensure target is part of chunk
                
                score = self.getSemanticSimilarity(chunk, target)
                
                if score > bestScore or (bestScore < 0 and windowSize == minExpand):
                    bestScore = score
                    bestChunk = chunk
                    bestSpan = (start + startIdx, end + startIdx)

        self.includedChecks[bestSpan[0]: bestSpan[1]] = True
        return bestChunk if bestScore > minSim else ""
    
    def retrieveFromDoc(self, searchString:str, topK:int, minSim:float, smoothingWindowSize:int=5):
        
        searchEmbedding = torch.tensor(self.getEmbedding(searchString)).unsqueeze(0)
        simScores = np.zeros(self.sentences.shape[0], dtype=float)

        for i in range(len(self.sentences)):
            sentenceEmbedding = torch.tensor(self.embeddings[i, :]).unsqueeze(0)
            simScores[i] = cosine_similarity(searchEmbedding, sentenceEmbedding).item()
        
        # SMOOTHEN
        # Smoothen out sim scores using moving average || [try gaussian smoothing too]
        simScores = np.convolve(simScores, np.ones(smoothingWindowSize)/smoothingWindowSize, mode='same')
        
        # THRESHOLDING & SELECTION
        selectedIdx = np.argsort(simScores)[-topK:]
        finalInfo = []
        for idx in selectedIdx:
            chunk = self._chunkify(targetIdx=idx, currentSim=simScores[idx], minSim=minSim)
            if chunk:
                finalInfo.append(chunk)
        # selection = self.sentences[simScores >= minSimilarity]
        return "\n".join((self.sentences[self.includedChecks]).tolist()) 
        # finalString = "\n".join(self.sentences[self.includedChecks]) 
        # return self.deduplicate(finalString)
    
    def deduplicate(self, text: str) -> str:
        """
        Remove duplicate sentences from the retrieved text chunks.
        
        Args:
            text (str): The concatenated text chunks from retrieveFromDoc
            
        Returns:
            str: The deduplicated text with unique sentences only
        """
        if not text:
            return ""
            
        # Split the text into chunks (if any)
        chunks = text.split('\n')
        
        # Process each chunk to extract sentences
        all_sentences = []
        for chunk in chunks:
            sentences = sent_tokenize(chunk)
            all_sentences.extend(sentences)
        
        # Use a set to track unique sentences (preserving order)
        unique_sentences = []
        seen = set()
        
        for sentence in all_sentences:
            # Normalize the sentence by removing extra whitespace
            normalized = ' '.join(sentence.split())
            if normalized not in seen and normalized:
                seen.add(normalized)
                unique_sentences.append(sentence)
        
        # Recombine the unique sentences
        return ' '.join(unique_sentences)





# FOR TESTING PURPOSE
if __name__ == "__main__":
    embedder = SemanticEmbedder()
    sent1 = "Me and my friends ate eight bananas and five apples towards the end of the party"
    sent2 = "eight bananas and five apples were eaten by some of my friends along with me after the party had ended"

    embedder.load()
    embedder.getSemanticSimilarity(sent1, sent2, show=True)
    # Test the RAG functionality
    sample_document = """
    Deep learning is part of a broader family of machine learning methods based on artificial neural networks.
    Learning can be supervised, semi-supervised or unsupervised.
    Deep learning architectures such as deep neural networks, deep belief networks, recurrent neural networks and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, bioinformatics, drug design, medical image analysis, material inspection and board game programs, where they have produced results comparable to and in some cases surpassing human expert performance.
    Artificial neural networks (ANNs) were inspired by information processing and distributed communication nodes in biological systems.
    ANNs have various differences from biological brains. Specifically, neural networks tend to be static and symbolic, while the biological brain of most living organisms is dynamic (plastic) and analog.
    The adjective "deep" in deep learning refers to the use of multiple layers in the network. Early work showed that a linear perceptron cannot be a universal classifier, but that a network with a nonpolynomial activation function with one hidden layer of unbounded width can.
    Deep learning is a modern variation which is concerned with an unbounded number of layers of bounded size, which permits practical application and optimized implementation, while retaining theoretical universality under mild conditions.
    """
    
    search_query = "What are artificial neural networks and how do they relate to deep learning?"
    
    # Initialize RAG with document
    embedder.initRAG(sample_document)
    print(f"NumSentences: {len(embedder.sentences)}")
    
    # Test retrieval
    print("\nTesting RAG Retrieval:")
    retrieved_info = embedder.retrieveFromDoc(
        searchString=search_query,
        topK=2,
        minSim=0.6,
        smoothingWindowSize=3
    )
    
    print(f"Query: {search_query}")
    print("Retrieved Information:")
    print(retrieved_info)
    
    # Test chunk extraction
    print("\nTesting Chunk Extraction:")
    sentences = embedder.sentences
    target_idx = 3  # Index for "Artificial neural networks..."
    chunk = embedder._chunkify(targetIdx=target_idx, currentSim=0.7, minSim=0.6)
    print(f"Extracted chunk containing sentence index {target_idx}:")
    print(chunk)
    
    embedder.unload()