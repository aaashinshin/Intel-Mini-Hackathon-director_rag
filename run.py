from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
import json
import os

from transformers import AutoTokenizer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM, RtnConfig
from transformers import TextStreamer


def run_model(question, history_data):

    device = "cpu"

    model_name = "qwen/Qwen1.5-7B-Chat"
    generate_kwargs = dict(do_sample=False, temperature=0.1, num_beams=1)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    emb_model_name = "bge-m3"
    embeddings = HuggingFaceBgeEmbeddings(model_name=emb_model_name)

    load_db = FAISS.load_local("faiss_index_database_director", embeddings, allow_dangerous_deserialization=True)

    faiss_retriever = load_db.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 4, 'fetch_k': 10, 'lambda_mult': 1}
    )

    if len(history_data) > 2:
        history_data.pop(0)
    history = '\n'.join(history_data)

    docs = faiss_retriever.get_relevant_documents(
        query=question
    )
    with open('director.json', 'r', encoding='utf-8') as f:
        Document = json.load(f)

    all_content = []
    content_set = set()
    for doc in docs:
        doc_metadata = doc.metadata
        for doc in Document:
            if doc_metadata == doc['metadata'] and doc['page_content'] not in content_set:
                all_content.append(doc['page_content'])
                content_set.add(doc['page_content'])

    content = '\n'.join(all_content)
    prompt = f'''
    身份:
    作为专业的TECS Director云管理平台运维专家，你的任务是根据《TECS Director运维手册》相关知识和标准，对运维问题进行精确解答。

    能力:
    - 仔细分析题目所提供的文档数据，确保理解和利用文档中的每一个细节。
    - 基于文档内容提供准确无误的答案，不做任何超出文档范围的假设。

    细节:
    - 细致审查文档中的所有信息，确保答案涵盖了所有相关的点。
    - 对于文档中的任何细节，即使是看似次要的信息，也要给予充分的关注。
    - 若文档信息不完整，指出需要哪些附加信息来准确回答问题。

    [任务]
    你的任务是根据所给的《TECS Director运维手册》文档数据，回答相关的运维问题。请确保你的回答严格基于文档中的信息，不要引入外部知识或假设。在你提供答案之后，请再次检查以确认答案的准确性，并确保完全符合文档所述。

    题目：[{question}]
    文本：[{content}]
    历史对话：[{history}]
    '''

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    woq_config = RtnConfig(bits=4, compute_dtype="int8")
    woq_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=woq_config,
    )

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    gen_ids = woq_model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        streamer=streamer,
        **generate_kwargs,
    )
    gen_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, gen_ids)
    ]
    gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]

    new_conversation = f'''
    user:{question}
    assistant:{gen_text}
    '''
    history_data.append(new_conversation)

    return gen_text, history_data