import re
import torch
import chromadb
import time
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

chromadbpath = './chromadb' #путь хранения базы данных всех книг
rerankerModel = 'Qwen/Qwen3-Reranker-0.6B' #реранкер модель
embeddingsModel = 'paraphrase-multilingual-MiniLM-L12-v2' #векторная модель
llmModel = 'Qwen/Qwen3-1.7B'

def log(string):
    print(string)

#embedding
MDA = 200 #Минимальная длина абзаца, прога режет текст на куски по обзацам, по срезам крч \n и инода эти срезы могут быть пипец маленькими, там когда личная речь идет или чет такое, так что было принято топовое решение объединять его со следующими кусками, так вот этот параметр задает сколько символов минимально будет содержать 1 кусочке, от ттуда и название MDA Minimal Dlina Abzac
emb_count = 20 #сколько ответов будет давать сравнение по косинусам векторов


#Qwen3-Reranker-0.6B
max_reranker_length = 8192
rerank_count = 3 #на основание скольки элементов llm будет давать ответ, то есть сколько реранкер будет давать вариантов
reranker_tokenizer = AutoTokenizer.from_pretrained(rerankerModel, padding_side='left')
reranker_model = AutoModelForCausalLM.from_pretrained(rerankerModel).eval()
token_false_id = reranker_tokenizer.convert_tokens_to_ids('no')
token_true_id = reranker_tokenizer.convert_tokens_to_ids('yes')
prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = reranker_tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = reranker_tokenizer.encode(suffix, add_special_tokens=False)
start = float

prePrompt = (
    "You are a strict factual assistant. You will be given numbered excerpts from books, "
    "each labeled with a filename and paragraph number. "
    "Answer the user's question using ONLY the information from these excerpts. "
    "Your answer must be 1-2 sentences maximum. "
    "At the end of your answer, always cite your source in this exact format: "
    "(Книга: [filename], Абзац: [abzac]). "
    "If the answer spans multiple excerpts, cite all of them. "
    "Do NOT add headers, bullet points, lists, or any extra commentary. "
    "Do NOT use your own knowledge or opinions — only what is written in the excerpts. "
    "If the answer is not present in the excerpts, respond only with: "
    "'Эта информация здесь отсутствует.'"
)
log('Запуск llm')
tokenizer = AutoTokenizer.from_pretrained(llmModel)
model = AutoModelForCausalLM.from_pretrained(
    llmModel,
    torch_dtype="auto",
    device_map="auto"
)
log('llm запущенна')

log('Запуск модели векторизации')
embs = SentenceTransformer(embeddingsModel)
log('Модели векторизации Запущена')

def format_instruction(instruction, WhatNeedSearch, doc):
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=WhatNeedSearch, doc=doc)
    return output

def process_inputs(pairs):
    inputs = reranker_tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_reranker_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = reranker_tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_reranker_length)
    for key in inputs:
        inputs[key] = inputs[key].to(reranker_model.device)
    return inputs

@torch.no_grad()
def compute_logits(inputs):
    batch_scores = reranker_model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores

def if_text_VERYSMALL(text:list):
    output = []
    element = ''
    for abzazc in text:
        if len(abzazc) > MDA:
            if len(element) > 1:
                element += abzazc
                output.append(element)
                element = ''
            else:
                output.append(abzazc)
        else:
            element += abzazc
    return output

def TTE(knigo: str,file_name: str):
    knigo = re.sub(r"[^\w .\n]", " ", knigo)
    knigo = re.sub(r' {2,}', ' ', knigo)
    knigo = knigo.split("\n") #НЕ ЗАБУТЬ ДОБАВИТЬ ДРОБИТЕЛЬ ЕСЛИ ЧАНК БОЛЬШЕ 500 СИМВОЛОВ ПРОСТО ПО ПОЛАМ НА ПОХУЙ
    knigo = if_text_VERYSMALL(knigo)
    knigores = []
    b = 1
    for chunk in knigo:
        vv = {
            "text": chunk,
            "textemb" : embs.encode(chunk).tolist(),
            "dataabout":{
                "filename": file_name.replace('.txt', ''),
                "abzac": b
            }
        }
        b+=1
        knigores.append(vv)
    return knigores

def chromadbcreate():
    bookcol = chromadb.PersistentClient(chromadbpath)
    return bookcol.get_or_create_collection("books")

def chromadbadd(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        filetext = file.read()
    db = chromadbcreate()
    chunks = TTE(filetext, file_path)
    chunks = [c for c in chunks if len(c["text"].strip()) > 20]
    allt,alle,allid,allm = [],[],[],[]
    for i in chunks:
        allt.append(i["text"])
        alle.append(i["textemb"])
        allid.append(file_path.replace('.txt', '')+str(i["dataabout"]["abzac"]))
        allm.append(i["dataabout"])
    db.add(
        documents=allt,
        embeddings=alle,
        ids=allid,
        metadatas=allm
    )

def reranksearch(WhatNeedSearch:str):
    WhatNeedSearcg_emb = embs.encode(WhatNeedSearch).tolist()
    db = chromadbcreate()
    top20search = db.query(
    query_embeddings=[WhatNeedSearcg_emb],
    n_results=emb_count
)
    documents = top20search["documents"][0]
    metadatas = top20search["metadatas"][0]
    prefix_abzac_sufix = []
    for abzac in documents:
        prefix_abzac_sufix.append(format_instruction(instruction=None, WhatNeedSearch=WhatNeedSearch, doc=abzac))
    prepare = process_inputs(prefix_abzac_sufix)
    score = compute_logits(prepare)
    rustresult = list(zip(documents, metadatas, score))
    rustresult.sort(key=lambda x: x[2], reverse=True)
    scoresortotrivki = rustresult[:rerank_count]
    otrivki_esli_bolshe_40 = []
    for otrivok in scoresortotrivki:
        if otrivok[2]>0.4:
            otrivki_esli_bolshe_40.append(otrivok)
    result = []
    for doc, meta, score in otrivki_esli_bolshe_40:
        result.append({
            "document": doc,
            "metadata": meta,
            "score": score
                       })
    return result

def LLMsearch(SearchQuery:str, DOCSSWAT):
    if len(DOCSSWAT) == 0:
        return 'Эта информация здесь отсутствует'
    context = ""
    for i, item in enumerate(DOCSSWAT):
        context += (
            f"[Отрывок {i+1}] "
            f"Книга: {item['metadata']['filename']} | "
            f"Абзац: {item['metadata']['abzac']}:\n"
            f"{item['document']}\n\n"
        )
    messages = [
        {"role": "system", "content": prePrompt},
        {"role": "user", "content": f"Отрывки из книги:\n{context}\nВопрос: {SearchQuery}"}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return content