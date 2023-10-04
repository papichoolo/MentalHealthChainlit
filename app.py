import chainlit as cl

import os
huggingfacehub_api_token = os.environ['HUGGINGFACEHUB_API_TOKEN']

from langchain import HuggingFaceHub, PromptTemplate, LLMChain

repo_id = "meta-llama/Llama-2-7b-chat-hf"
llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token, 
                     repo_id=repo_id, 
                     model_kwargs={"temperature":0.6, "max_new_tokens":2000})

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<>\n", "\n<>\n\n"
template = sys_msg = "" + B_SYS + """
You are a mental health adviser. You will have a conversation with an individual looking for guidance and advice on managing their emotions, stress, anxiety and other mental health issues. You should use your knowledge of cognitive behavioral therapy, meditation techniques, mindfulness practices, and other therapeutic methods in order to create strategies that the individual can implement in order to improve their overall wellbeing after listening to your patient like a friend. You will stop after the Human is done talking




Here is the latest conversation between You and User \n
 {chat_history} """ + E_SYS
instruction = B_INST + " You will listen to this User and give mental support. Remember to only be the Mental Health Adviser " + E_INST
human_msg = instruction + "\nUser: {human_input} \n Mental Health Adviser:"
template+=human_msg


@cl.on_chat_start
def main():
    # Instantiate the chain for that user session
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

    # Store the chain in the user session
    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain

    # Call the chain synchronously in a different thread
    res = await cl.make_async(llm_chain)(
        message, callbacks=[cl.LangchainCallbackHandler()]
    )

    # Do any post processing here

    # "res" is a Dict. For this chain, we get the response by reading the "text" key.
    # This varies from chain to chain, you should check which key to read.
    await cl.Message(content=res["text"]).send()
    return llm_chain