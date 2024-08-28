import copy
import warnings
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional
import os
os.system("pip install streamlit")

import streamlit as st
import torch
from FlagEmbedding import FlagReranker
from torch import nn
from transformers.generation.utils import (LogitsProcessorList,
                                           StoppingCriteriaList)
from transformers.utils import logging
from langchain.vectorstores import  FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM  # isort: skip
# from hougegpt import (InternLM_LLM,
#                       retrieve_context_per_question,
#                       keep_only_relevant_content,
#                       answer_question_from_context,
#                       get_answer_prompt)

logger = logging.get_logger(__name__)




from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
import torch
from langchain.vectorstores import  FAISS
from langchain.prompts import PromptTemplate


base_path = './final_model'
os.system(f'git clone https://code.openxlab.org.cn/bob12/hougeGPT.git {base_path}')
embedding_path = './bce-embedding-base_v1'
os.system(f'git clone https://www.modelscope.cn/AI-ModelScope/bge-large-zh-v1.5.git {embedding_path}')
reranker_path = './bce-reranker-base_v1'
os.system(f'git clone https://www.modelscope.cn/AI-ModelScope/bge-reranker-v2-m3.git {reranker_path}')
os.system(f'cd {base_path} && git lfs pull')
os.system(f'cd {embedding_path} && git lfs pull')
os.system(f'cd {reranker_path} && git lfs pull')

model_name_or_path = "final_model"
chunks_vector_store = "./chunks_vector_store"
chapter_summaries_vector_store = "./chapter_summaries_vector_store"
embedding_path="bce-embedding-base_v1"
bge_embedding_path="bce-embedding-base_v1"
rerank_path = "bce-reranker-base_v1"
beg_chunks_vector_store = "./beg_chunks_vector_store"
beg_chapter_summaries_vector_store = "./beg_chapter_summaries_vector_store"




# model_name_or_path = "/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b"
# chunks_vector_store = "./chunks_vector_store"
# chapter_summaries_vector_store = "./chapter_summaries_vector_store"
# embedding_path="/root/bce-embedding-base_v1"
# bge_embedding_path="/root/bge-large-zh-v1.5"
# rerank_path = "/root/bge-reranker-v2-m3"
# beg_chunks_vector_store = "./beg_chunks_vector_store"
# beg_chapter_summaries_vector_store = "./beg_chapter_summaries_vector_store"


class InternLM_LLM(LLM):
    # 基于本地 InternLM 自定义 LLM 类
    tokenizer : AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, llm,tokenizer):
        # model_path: InternLM 模型路径
        # 从本地初始化模型
        super().__init__()
        print("正在从本地加载模型...")
        self.tokenizer = tokenizer
        self.model = llm
        self.model = self.model.eval()
        print("完成本地模型的加载")

    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):
        # 重写调用函数
        system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
        - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
        - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
        """
        
        messages = [(system_prompt, '')]
        response, history = self.model.chat(self.tokenizer, prompt , history=messages)
        return response
        
    @property
    def _llm_type(self) -> str:
        return "InternLM"
    
def escape_quotes(text):
  """Escapes both single and double quotes in a string.

  Args:
    text: The string to escape.

  Returns:
    The string with single and double quotes escaped.
  """
  return text.replace('"', '\\"').replace("'", "\\'")

def retrieve_context_per_question(state,chunks_query_retriever,chapter_summaries_query_retriever):
    """
    Retrieves relevant context for a given question. The context is retrieved from the book chunks and chapter summaries.

    Args:
        state: A dictionary containing the question to answer.
    """
    # Retrieve relevant documents
    print("检索相关 chunks...")
    question = state["question"]
    docs = chunks_query_retriever.invoke(question)

    # Concatenate document content
    context = " ".join(doc.page_content for doc in docs)



    print("检索相关章节 summaries...")
    docs_summaries = chapter_summaries_query_retriever.invoke(state["question"])

    # Concatenate chapter summaries with citation information
    context_summaries = " ".join(
        f"{doc.page_content} (Chapter {doc.metadata['chapter']})" for doc in docs_summaries
    )

    # print("Retrieving relevant book quotes...")
    # docs_book_quotes = book_quotes_query_retriever.get_relevant_documents(state["question"])
    # book_qoutes = " ".join(doc.page_content for doc in docs_book_quotes)


    all_contexts = context + context_summaries #+ book_qoutes
    all_contexts = escape_quotes(all_contexts)

    return {"context": all_contexts, "question": question}




def keep_only_relevant_content(state,llm):
    """
    Keeps only the relevant content from the retrieved documents that is relevant to the query.

    Args:
        question: The query question.
        context: The retrieved documents.
        chain: The LLMChain instance.

    Returns:
        The relevant content from the retrieved documents that is relevant to the query.
    """



    class KeepRelevantContent(BaseModel):
        relevant_content: str = Field(description="The relevant content from the retrieved documents that is relevant to the query.")


    question = state["question"]
    context = state["context"]

    input_data = {
    "query": question,
    "retrieved_documents": context
    }
    print("LLM 处理相关 content...")

    keep_only_relevant_content_parser = JsonOutputParser(pydantic_object=KeepRelevantContent)

    #keep_only_relevant_content_llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=2000)
    keep_only_relevant_content_prompt_template = """您将收到一个查询：{query}和检索到的文档：{retrieved_documents}矢量存储。您需要过滤掉所有不提供有关{query}的重要信息的非相关信息。
    你的目标只是过滤掉不相关的信息。
    您可以删除与查询不相关的句子部分，或删除与查询无关的整个句子。
    不要添加任何不在检索到的文档中的新信息。
    输出过滤后的相关内容。
    """

    keep_only_relevant_content_prompt = PromptTemplate(
    template=keep_only_relevant_content_prompt_template,
    input_variables=["query", "retrieved_documents"],
    )
    
    keep_only_relevant_content_chain = keep_only_relevant_content_prompt | llm #| keep_only_relevant_content_parser
    

    print("--------------------")
    output = keep_only_relevant_content_chain.invoke(input_data)
    #print(output)
    relevant_content = output
    relevant_content = "".join(relevant_content)
    relevant_content = escape_quotes(relevant_content)

    return {"relevant_context": relevant_content, "context": context, "question": question}



def answer_question_from_context(state,llm):
    """
    Answers a question from a given context.

    Args:
        question: The query question.
        context: The context to answer the question from.
        chain: The LLMChain instance.

    Returns:
        The answer to the question from the context.
    """

    question_answer_cot_prompt_template = """ 
    Examples of Chain-of-Thought Reasoning

    Example 
    Context: 哈利正在读一本关于咒语的书。一个咒语可以让施法者在短时间内将人变成动物。另一个咒语可以使物体漂浮。
    第三个咒语在施法者魔杖的末端产生了一道亮光.
    Question: 根据上下文，如果哈利施展这些咒语，他能做什么?
    Reasoning Chain:
    上下文描述了三种不同的咒语
    第一个咒语允许将人暂时变成动物
    第二个咒语可以使物体漂浮
    第三个咒语会产生明亮的光线
    如果哈利施展这些咒语，他可以在一段时间内将某人变成动物，使物体漂浮，并创造出明亮的光源
    因此，根据上下文，如果哈利施展这些咒语，他可以改变人，使物体漂浮，照亮一个区域
    说明。

    对于下面的问题，首先展示你的逐步推理过程，在得出最终答案之前将问题分解为一连串的思考，然后给出你的答案，
    就像前面的例子一样。
    Context
    {context}
    Question
    {question}
    """


    question_answer_from_context_cot_prompt = PromptTemplate(
        template=question_answer_cot_prompt_template,
        input_variables=["context", "question"],
    )

    question_answer_from_context_cot_chain = question_answer_from_context_cot_prompt | llm

    question = state["question"]
    context = state["aggregated_context"] if "aggregated_context" in state else state["context"]

    input_data = {
    "question": question,
    "context": context
    }
    print("从相关内容中回答相关问题...")

    output = question_answer_from_context_cot_chain.invoke(input_data)
    print("完成从相关内容中回答相关问题")
    #print(output)
    answer = output
    #print(f'answer before checking hallucination: {answer}')
    return {"answer": answer, "context": context, "question": question}

def get_query_classification(llm,stat):
    pass


def get_answer_prompt(state):
    question_answer_cot_prompt_template1 = """ 
    Examples of Chain-of-Thought Reasoning

    Example 
    Context: 哈利正在读一本关于咒语的书。一个咒语可以让施法者在短时间内将人变成动物。另一个咒语可以使物体漂浮。
    第三个咒语在施法者魔杖的末端产生了一道亮光.
    Question: 根据上下文，如果哈利施展这些咒语，他能做什么?
    Reasoning Chain:
    上下文描述了三种不同的咒语
    第一个咒语允许将人暂时变成动物
    第二个咒语可以使物体漂浮
    第三个咒语会产生明亮的光线
    如果哈利施展这些咒语，他可以在一段时间内将某人变成动物，使物体漂浮，并创造出明亮的光源
    因此，根据上下文，如果哈利施展这些咒语，他可以改变人，使物体漂浮，照亮一个区域
    说明。

    对于下面的问题，首先展示你的逐步推理过程，在得出最终答案之前将问题分解为一连串的思考，然后给出你的答案，
    就像前面的例子一样。
    Context
    {context}
    Question
    {question}
    """
    question_answer_cot_prompt_template = """ 
    你是一个小说家，请根据下面的Context内容回答Question。
    Context
    {context}
    Question
    {question}
    """

    if "关系" in state["question"]:
        question_answer_cot_prompt_template = question_answer_cot_prompt_template1

    question_answer_from_context_cot_prompt = PromptTemplate(
        template=question_answer_cot_prompt_template,
        input_variables=["context", "question"],
    )

    question = state["question"]
    context = state["aggregated_context"] if "aggregated_context" in state else state["context"]

    input_data = {
    "question": question,
    "context": context
    }
    print("从相关内容中回答相关问题...")

    return question_answer_from_context_cot_prompt.invoke(input_data).text


def get_rank(query,relevate_document,reranker,top_n=1):
    querys =[[query,i.page_content] for i in relevate_document]
    scores = reranker.compute_score(querys, normalize=True)
    querys_documents = list(zip(scores,relevate_document))
    querys_documents = sorted(querys_documents,key=lambda x:x[0],reverse=True)
    scores,relevate_document = zip(*querys_documents)
    return scores[:top_n],relevate_document[:top_n]

def beg_retrieve_context_per_question(state,beg_chunks_query_retriever,beg_chapter_summaries_query_retriever,beg_rerank=None):
    """
    Retrieves relevant context for a given question. The context is retrieved from the book chunks and chapter summaries.

    Args:
        state: A dictionary containing the question to answer.
    """
    # Retrieve relevant documents


    print("检索相关 chunks...")
    question = state["question"]
    docs = beg_chunks_query_retriever.invoke(question)

    print("检索相关章节 summaries...")
    docs_summaries = beg_chapter_summaries_query_retriever.invoke(state["question"])

    print("reranks ...")

    scores,rerank_document = get_rank(question,docs+docs_summaries,beg_rerank)
    all_contexts = " ".join(doc.page_content for doc in rerank_document)
    all_contexts = escape_quotes(all_contexts)

    return {"context": all_contexts, "question": question}




@dataclass
class GenerationConfig:
    # this config is used for chat to provide more diversity
    max_length: int = 8192
    top_p: float = 0.75
    temperature: float = 0.1
    do_sample: bool = True
    repetition_penalty: float = 1.000


@torch.inference_mode()
def generate_interactive(
    model,
    tokenizer,
    prompt,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor],
                                                List[int]]] = None,
    additional_eos_token_id: Optional[int] = None,
    **kwargs,
):
    #print("prompt",prompt)
    print("开始生成")
    inputs = tokenizer([prompt], padding=True, return_tensors='pt')
    input_length = len(inputs['input_ids'][0])
    for k, v in inputs.items():
        inputs[k] = v.cuda()
    input_ids = inputs['input_ids']
    _, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]
    if generation_config is None:
        generation_config = model.generation_config
    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)
    bos_token_id, eos_token_id = (  # noqa: F841  # pylint: disable=W0612
        generation_config.bos_token_id,
        generation_config.eos_token_id,
    )
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    if additional_eos_token_id is not None:
        eos_token_id.append(additional_eos_token_id)
    has_default_max_length = kwargs.get(
        'max_length') is None and generation_config.max_length is not None
    if has_default_max_length and generation_config.max_new_tokens is None:
        warnings.warn(
            f"Using 'max_length''s default ({repr(generation_config.max_length)}) \
                to control the generation length. "
            'This behaviour is deprecated and will be removed from the \
                config in v5 of Transformers -- we'
            ' recommend using `max_new_tokens` to control the maximum \
                length of the generation.',
            UserWarning,
        )
    elif generation_config.max_new_tokens is not None:
        generation_config.max_length = generation_config.max_new_tokens + \
            input_ids_seq_length
        if not has_default_max_length:
            logger.warn(  # pylint: disable=W4902
                f"Both 'max_new_tokens' (={generation_config.max_new_tokens}) "
                f"and 'max_length'(={generation_config.max_length}) seem to "
                "have been set. 'max_new_tokens' will take precedence. "
                'Please refer to the documentation for more information. '
                '(https://huggingface.co/docs/transformers/main/'
                'en/main_classes/text_generation)',
                UserWarning,
            )

    if input_ids_seq_length >= generation_config.max_length:
        input_ids_string = 'input_ids'
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids_seq_length}, "
            f"but 'max_length' is set to {generation_config.max_length}. "
            'This can lead to unexpected behavior. You should consider'
            " increasing 'max_new_tokens'.")

    # 2. Set generation parameters if not already defined
    logits_processor = logits_processor if logits_processor is not None \
        else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None \
        else StoppingCriteriaList()

    logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )

    stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config,
        stopping_criteria=stopping_criteria)
    logits_warper = model._get_logits_warper(generation_config)

    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    scores = None
    while True:
        model_inputs = model.prepare_inputs_for_generation(
            input_ids, **model_kwargs)
        # forward pass to get next token
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # sample
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        if generation_config.do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=False)
        unfinished_sequences = unfinished_sequences.mul(
            (min(next_tokens != i for i in eos_token_id)).long())

        output_token_ids = input_ids[0].cpu().tolist()
        #print("decode",tokenizer.decode(output_token_ids))
        output_token_ids = output_token_ids[input_length:]
        for each_eos_token_id in eos_token_id:
            if output_token_ids[-1] == each_eos_token_id:
                output_token_ids = output_token_ids[:-1]
        response = tokenizer.decode(output_token_ids)

        yield response
        # stop when each sentence is finished
        # or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(
                input_ids, scores):
            print("break")
            break
        #print("ggsddf",response)


def on_btn_click():
    del st.session_state.messages


@st.cache_resource
def load_model():
    model = (AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                  trust_remote_code=True).to(
                                                      torch.bfloat16).cuda())
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                              trust_remote_code=True)
    return model, tokenizer

@st.cache_resource
def load_rag(embedding_path="/root/bce-embedding-base_v1" ,chunks_vector_store="chunks_vector_store",chapter_summaries_vector_store='chapter_summaries_vector_store'):
    embeddings_model = HuggingFaceEmbeddings(model_name=embedding_path)
    chunks_vector =  FAISS.load_local(chunks_vector_store, embeddings_model, allow_dangerous_deserialization=True)
    chapter_summaries_vector=  FAISS.load_local(chapter_summaries_vector_store, embeddings_model, allow_dangerous_deserialization=True)
    return chunks_vector, chapter_summaries_vector,embeddings_model

@st.cache_resource
def load_rerank():
    
    beg_rerank = FlagReranker(rerank_path, use_fp16=True)
    return beg_rerank

def prepare_generation_config():
    with st.sidebar:
        max_length = st.slider('Max Length',
                               min_value=8,
                               max_value=32768,
                               value=4096)
        top_p = st.slider('Top P', 0.0, 1.0, 0.75, step=0.01)
        temperature = st.slider('Temperature', 0.0, 1.0, 0.1, step=0.01)
        st.button('Clear Chat History', on_click=on_btn_click)

    generation_config = GenerationConfig(max_length=max_length,
                                         top_p=top_p,
                                         temperature=temperature)

    return generation_config


user_prompt = '<|im_start|>user\n{user}<|im_end|>\n'
robot_prompt = '<|im_start|>assistant\n{robot}<|im_end|>\n'
cur_query_prompt = '<|im_start|>user\n{user}<|im_end|>\n\
    <|im_start|>assistant\n'


def combine_history(prompt):
    messages = st.session_state.messages
    meta_instruction = ('')
    total_prompt = f"<s><|im_start|>system\n{meta_instruction}<|im_end|>\n"
    # for message in messages:
    #     cur_content = message['content']
    #     if message['role'] == 'user':
    #         cur_prompt = user_prompt.format(user=cur_content)
    #     elif message['role'] == 'robot':
    #         cur_prompt = robot_prompt.format(robot=cur_content)
    #     else:
    #         raise RuntimeError
    #     total_prompt += cur_prompt
    total_prompt = total_prompt + cur_query_prompt.format(user=prompt)
    return total_prompt


def main():
    # torch.cuda.empty_cache()
    print('load model begin.')
    model, tokenizer = load_model()
    print('load model end.')
    BEG = True
    rank_model = None

    if not BEG:
        print('begin bce')
        chunks_vector ,chapter_summaries_vector,embedding= load_rag()
        print('end bce')
    else:
        print('begin bge')
        chunks_vector ,chapter_summaries_vector,embedding= load_rag(bge_embedding_path,chunks_vector_store=beg_chunks_vector_store,chapter_summaries_vector_store=beg_chapter_summaries_vector_store)
        rank_model = load_rerank()
        print('end bge')

    chunks_query_retriever = chunks_vector.as_retriever(search_kwargs={"k": 1})     
    chapter_summaries_query_retriever = chapter_summaries_vector.as_retriever(search_kwargs={"k": 1})


    st.title('HougeGPT')
    llm = InternLM_LLM(model,tokenizer)


    generation_config = prepare_generation_config()

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message['role'], avatar=message.get('avatar')):
            st.markdown(message['content'])

    # Accept user input
    if prompt := st.chat_input('有啥事找俺猴哥?'):
        # Display user message in chat message container
        with st.chat_message('user'):
            st.markdown(prompt)
        real_prompt = combine_history(prompt)
        # Add user message to chat history
        st.session_state.messages.append({
            'role': 'user',
            'content': prompt,
        })

        if not BEG:
            sta = retrieve_context_per_question({"question":prompt},chunks_query_retriever,chapter_summaries_query_retriever)
        else:
            sta = beg_retrieve_context_per_question({"question":prompt},beg_chunks_query_retriever=chunks_query_retriever,beg_chapter_summaries_query_retriever=chapter_summaries_query_retriever,beg_rerank=rank_model)
        keep_only_relevant = keep_only_relevant_content(sta,llm)
        prompt = get_answer_prompt(keep_only_relevant)
        #answer_question_from_context(keep_only_relevant,llm)
        prompt = prompt[:2048]
        real_prompt = combine_history(prompt)
        with st.chat_message('robot'):
            print("chat begin",prompt)
            message_placeholder = st.empty()
            for cur_response in generate_interactive(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=real_prompt,
                    additional_eos_token_id=92542,
                    **asdict(generation_config),
            ):
                # Display robot response in chat message container
                message_placeholder.markdown(cur_response + '▌')
            message_placeholder.markdown(cur_response)
        # Add robot response to chat history
        st.session_state.messages.append({
            'role': 'robot',
            'content': cur_response,  # pylint: disable=undefined-loop-variable
        })
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
