from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import sent_tokenize
from torch.nn.functional import cosine_similarity
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
                "name": "punkt"
            }
        ):
        # Semantic Checker to load
        self.semanticChecker = {
            'path': f"Assets/Semantic-Checkers/{semanticChecker['folder']}",
            'name': semanticChecker['name']
        }
        self.sentenceTokenizer = {
            'path': f"Assets/{sentenceTokenizer['folder']}/",
            'name': sentenceTokenizer['name']
        } 
        self.DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
        
        self.TOKENIZER:AutoTokenizer = None 
        self.MODEL:AutoModel = None
        self.LOADED = False
        
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
            nltk.download(info_or_id=self.sentenceTokenizer['name'], download_dir=self.sentenceTokenizer['path'])
            print("Download Complete\n")
            nltk.data.path.append(self.sentenceTokenizer['path'])

    
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

    def getSentenceEmbedding(self, sentence:str):
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
        return (sum_embeddings / sum_mask)[0].numpy()
    
    def getSemanticSimilarity(self, sent1:str, sent2:str, show:bool = False):
        embedding1 = torch.tensor(self.getSentenceEmbedding(sent1)).unsqueeze(0)
        embedding2 = torch.tensor(self.getSentenceEmbedding(sent2)).unsqueeze(0)
        sim = cosine_similarity(embedding1, embedding2).item()
        if show:
            print(f"\n\t\tSEMANTIC SIMILIARITY = {sim: 0.4}")
            print(f"\t\t|---> Sentence 1: {sent1}")
            print(f"\t\t|---> Sentence 2: {sent2}")
        return sim

# FOR TESTING PURPOSE
if __name__ == "__main__":
    embedder = SemanticEmbedder()
    sent1 = "Me and my friends ate 8 bananas and 5 apples towards the end of the party"
    sent2 = "8 bananas and 5 apples were eaten by some of my friends along with me after the party had ended"

    embedder.load()
    embedder.getSemanticSimilarity(sent1, sent2, show=True)
    embedder.unload()