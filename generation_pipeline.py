from model_handler import ModelHandler
from retrieve import Retriever
from transformers import pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize retriever
retriever = Retriever()

model_name = "meta-llama/Llama-3.2-3B-Instruct"
m = ModelHandler(model_name, quantize=True)
model, tokenizer = m.load_model()

tokenizer.chat_template = """{{- bos_token -}}
{%- if custom_tools is defined -%}
	{%- set tools = custom_tools -%}
{%- endif -%}
{%- if not tools_in_user_message is defined -%}
	{%- set tools_in_user_message = true -%}
{%- endif -%}
{%- if not date_string is defined -%}
	{%- if strftime_now is defined -%}
		{%- set date_string = strftime_now("%d %b %Y") -%}
	{%- else -%}
		{%- set date_string = "26 Jul 2024" -%}
	{%- endif -%}
{%- endif -%}
{%- if not tools is defined -%}
	{%- set tools = none -%}
{%- endif -%}
{#  This block extracts the system message, so we can slot it into the right place.  #}
{%- if messages[0]["role"] == "system" -%}
	{%- set system_message = messages[0]["content"] | trim -%}
	{%- set messages = messages[1:] -%}
{%- else -%}
	{%- set system_message = "" -%}
{%- endif -%}
{#  System message  #}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" -}}
{%- if tools is not none -%}
	{{- "Environment: ipython\n" -}}
{%- endif -%}
{{- "Cutting Knowledge Date: December 2023\n" -}}
{{- "Today Date: " + date_string + "\n\n" -}}
{%- if tools is not none and not tools_in_user_message -%}
	{{- "You have access to the following functions. To call a function, please respond with JSON for a function call." -}}
	{{- "Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}." -}}
	{{- "Do not use variables.\n\n" -}}
	{%- for t in tools -%}
		{{- t | tojson(indent=4) -}}
		{{- "\n\n" -}}
	{%- endfor -%}
{%- endif -%}
{{- system_message -}}
{{- "<|eot_id|>" -}}
{#  Custom tools are passed in a user message with some extra guidance  #}
{%- if tools_in_user_message and not tools is none -%}
	{#  Extract the first user message so we can plug it in here  #}
	{%- if messages | length != 0 -%}
		{%- set first_user_message = messages[0]["content"] | trim -%}
		{%- set messages = messages[1:] -%}
	{%- else -%}
		{{- raise_exception("Cannot put tools in the first user message when there's no first user message!") -}}
	{%- endif -%}
	{{- "<|start_header_id|>user<|end_header_id|>\n\n" -}}
	{{- "Given the following functions, please respond with a JSON for a function call " -}}
	{{- "with its proper arguments that best answers the given prompt.\n\n" -}}
	{{- "Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}." -}}
	{{- "Do not use variables.\n\n" -}}
	{%- for t in tools -%}
		{{- t | tojson(indent=4) -}}
		{{- "\n\n" -}}
	{%- endfor -%}
	{{- first_user_message + "<|eot_id|>" -}}
{%- endif -%}
{%- for message in messages -%}
	{%- if not (message.role == "ipython" or message.role == "tool" or "tool_calls" in message) -%}
		{{- "<|start_header_id|>" + message["role"] + "<|end_header_id|>\n\n" + message["content"] | trim + "<|eot_id|>" -}}
	{%- elif "tool_calls" in message -%}
		{%- if not (message.tool_calls | length == 1) -%}
			{{- raise_exception("This model only supports single tool-calls at once!") -}}
		{%- endif -%}
		{%- set tool_call = message.tool_calls[0].function -%}
		{{- "<|start_header_id|>assistant<|end_header_id|>\n\n" -}}
		{{- "{\"name\": \"" + tool_call.name + "\", " -}}
		{{- "\"parameters\": " -}}
		{{- tool_call.arguments | tojson -}}
		{{- "}" -}}
		{{- "<|eot_id|>" -}}
	{%- elif message.role == "tool" or message.role == "ipython" -%}
		{{- "<|start_header_id|>ipython<|end_header_id|>\n\n" -}}
		{%- if message.content is mapping or message.content is iterable -%}
			{{- message.content | tojson -}}
		{%- else -%}
			{{- message.content -}}
		{%- endif -%}
		{{- "<|eot_id|>" -}}
	{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
	{{- "<|start_header_id|>assistant<|end_header_id|>\n\n" -}}
{%- endif -%}"""

hf_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.3,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    return_full_text=False
)

lc_pipe = HuggingFacePipeline(pipeline=hf_pipe)

template = """
You are an interview preparation assistant. Use the provided context to answer the user's question accurately and concisely.

Context:
{context}

Question:
{question}

Answer:
"""
prompt = PromptTemplate.from_template(template)

chain = prompt | lc_pipe | StrOutputParser()

def generate_answer(question: str, top_k: int = 5) -> dict:
    """
    Given a user question, retrieves relevant context and returns the LLaMA-generated answer.
    
    Args:
        question: User's question about invoices
        top_k: Number of chunks to retrieve
    
    Returns:
        dict with 'answer', 'sources', and 'retrieved_chunks'
    """
    # Retrieve relevant chunks
    retrieved = retriever.retrieve(question, top_k=top_k)
    
    # Combine chunks into context
    context = "\n\n".join([
        f"[Source: {chunk['source']}, Page: {chunk['page']}]\n{chunk['text']}"
        for chunk in retrieved
    ])
    
    # Generate answer
    answer = chain.invoke({"context": context, "question": question})
    
    # Extract unique sources
    sources = list(set([
        f"{chunk['source']} (Page {chunk['page']})"
        for chunk in retrieved
    ]))
    
    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": retrieved
    }

