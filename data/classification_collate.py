# Responsibility: Eric

import torch
from transformers import AutoTokenizer
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

# Returns a string starting from a random word boundary, pre-truncated to avoid                                                            
# tokenizing unnecessarily long text                                                           
def random_text_start(text, max_chars=3000):                                                                                               
    s = " " + text                                                                                                                         
    i = torch.randint(len(s), (1,)).item()                                                                                                 
    start = s.rfind(" ", 0, i) + 1                                                                                                         
    sample = (s[start:] + s[:start - 1]).strip()                                                                                           
    return sample[:max_chars]                   

# Collects individual documents into (text_A, text_B) pairs for classification based training via triplet mining, forming pairs, and tokenization
class ClassificationCollator:
    def __init__(self, model_name, max_length, prefix = ""):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)
        self.max_length = max_length
        self.prefix = prefix

    # Batch from MPerClassSampler (ensures at least m documents per author), each item consists of 'text' and 'author_int'
    def __call__(self, batch):
        texts = [x["text"] for x in batch]
        labels = torch.tensor([x["author_int"] for x in batch], dtype = torch.long)

        # Triplet mining to form positive and negative pairs, one (anchor, positive, negative) triplet per sample 
        a, p, n = lmu.get_random_triplet_indices(labels, t_per_anchor = 1)

        # Build positive pairs (same author, label = 1) and negative pairs (diff author, label = 0)
        pair_texts_a = []
        pair_texts_b = []
        pair_labels = []

        for i in range(len(a)):
            text_a = random_text_start(texts[a[i]])
            if self.prefix:
                text_a = self.prefix + text_a

            # Positive pair
            text_p = random_text_start(texts[p[i]])
            if self.prefix:
                text_p = self.prefix + text_p
            pair_texts_a.append(text_a)
            pair_texts_b.append(text_p)
            pair_labels.append(1)

            # Negative pair
            text_n = random_text_start(texts[n[i]])
            if self.prefix:
                text_n = self.prefix + text_n
            pair_texts_a.append(text_a)
            pair_texts_b.append(text_n)
            pair_labels.append(0)

        enc = self.tokenizer(
            pair_texts_a,
            pair_texts_b,
            add_special_tokens = True,
            truncation = "longest_first",
            max_length = self.max_length,
            padding = "max_length",
            return_tensors = "pt",
        )

        # inputs dict, labels tensor
        return enc, torch.tensor(pair_labels, dtype=torch.long)
    
# Collates pre-constructed pairs for val/test, tokenizes and returns labels. 
class ClassificationPairCollator:
    def __init__(self, model_name, max_length, prefix = ""):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)
        self.max_length = max_length
        self.prefix = prefix
        
    # Pre-truncates texts to avoid slow tokenization for long docs.
    def __call__(self, batch):                                                                                                                                                                        
        texts_a = [(self.prefix + x["text1"] if self.prefix else x["text1"])[:3000] for x in batch]                                        
        texts_b = [(self.prefix + x["text2"] if self.prefix else x["text2"])[:3000] for x in batch]                                        
                                            
        enc = self.tokenizer(                                                                                                              
            texts_a,                    
            texts_b,                                                                                                                       
            add_special_tokens = True,                                                                                                     
            truncation = "longest_first",                                                                                                  
            max_length = self.max_length,                                                                                                  
            padding = "max_length",         
            return_tensors = "pt",      
        )
                                                                                                                                            
        labels = torch.tensor([int(x["same"]) for x in batch], dtype = torch.long)
        # inputs dict, labels tensor                                                                                                       
        return enc, labels                  
