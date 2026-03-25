import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

os.makedirs('./models/reranker', exist_ok=True)
os.makedirs('./models/embeddings', exist_ok=True)
os.makedirs('./models/llm', exist_ok=True)

reranker_tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Reranker-0.6B', padding_side='left')
reranker_tokenizer.save_pretrained('./models/reranker')
reranker_model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-Reranker-0.6B').eval()
reranker_model.save_pretrained('./models/reranker')

embs = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
embs.save('./models/embeddings')

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B')
tokenizer.save_pretrained('./models/llm')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-1.7B', torch_dtype="auto", device_map="auto")
model.save_pretrained('./models/llm')