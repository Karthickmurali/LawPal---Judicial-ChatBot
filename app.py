from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
import torch
import chainlit as cl

isFirst = True

model_name = "BAAI/bge-large-en"
model_kwargs = {"device": 'cpu'}
encode_kwargs = {"normalize_embeddings":False}

embeddings = HuggingFaceBgeEmbeddings(
    model_kwargs = model_kwargs,
    model_name = model_name,
    encode_kwargs = encode_kwargs
)

url = "http://localhost:6333"
collection_name = "test_csv_db"

client = QdrantClient(
    url =url,
    prefer_grpc = False
    # prefer_grpc = True
)

db = Qdrant(
    client = client,
    embeddings = embeddings,
    collection_name = collection_name
)

query = "What is your Ocado Price Promise?"
docs = db.similarity_search_with_score(query=query, k=3)

# for doc, score in docs:
#     print({"score":score, "content":doc.page_content, "metadata":doc.metadata})
doc, score = docs[0]
print("##########################################################")
print({"score":score, "content":doc.page_content, "metadata":doc.metadata})
print("##########################################################")

custom_prompt_template = """
[INST]
You are an online customer service chatbot that resolves queries from customer of Ocado.com
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
## Context: 
{context}
------------------------------------------------------------------
## Question: 
{question}
Only return the helpful answer below and nothing else.
## Helpful answer:
[/INST]
"""
def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    global isFirst
    if isFirst:
        cpt = "<s>"+custom_prompt_template
        isFirst = False
    else:
        cpt = custom_prompt_template
    prompt = PromptTemplate(template=cpt,
                            input_variables=['context', 'question'])
    # prompt = PromptTemplate(template=custom_prompt_template,
    #                         input_variables=['context', 'question'])
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

def load_llm():
    # llm = CTransformers(
    #     model = r"D:\NK_Programming\ML\Examples\Llama2-Medical-Chatbot-main\Llama2-Medical-Chatbot-main\llama-2-7b-chat.ggmlv3.q8_0.bin",
    #     model_type="llama",
    #     max_new_tokens = 512,
    #     temperature = 0.75
    # )
    llm = CTransformers(
        model = r"D:\NK_Programming\AI-Models\mistral-7b-instruct-v0.1.Q5_K_M.gguf",
        model_type="mistral",
        max_new_tokens = 512,
        temperature = 0.35
    )
    return llm

def qa_bot():
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

#chainlit code
@cl.on_chat_start
async def start():
    isFirst = True
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to OCADO.com . What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$", res, "&&&&&&&&&&&&&&&&&&&&&&&&&&&&", sep="\n")
    answer = res["result"]
    sources = res["source_documents"]
    with open("res.txt", 'w') as f:
        f.write(str(res))
    with open("answer.txt", 'w') as f:
        f.write(str(answer))
    with open("sources.txt", 'w') as f:
        f.write(str(sources))
    rfcnc = "For more info..."
    if sources:
        for source in sources:
            rfcnc += f"\nRefer: " + str(source.metadata["LINKS"])
    else:
        rfcnc += "\nNo sources found"

    await cl.Message(content=rfcnc).send()
