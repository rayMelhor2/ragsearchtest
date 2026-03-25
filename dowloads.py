from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2').save('./models/paraphrase-multilingual-MiniLM-L12-v2')
AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B').save_pretrained('./models/Qwen3-1.7B')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-1.7B').save_pretrained('./models/Qwen3-1.7B')
AutoTokenizer.from_pretrained('Qwen/Qwen3-Reranker-0.6B', padding_side='left').save_pretrained('./models/Qwen3-Reranker-0.6B')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-Reranker-0.6B').save_pretrained('./models/Qwen3-Reranker-0.6B')