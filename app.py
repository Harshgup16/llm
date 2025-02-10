from fastapi import FastAPI
from langserve import add_routes
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Medical LLM API",
    version="1.0",
    description="API for medical language model using LangServe"
)
# Initialize the model with direct token and task
llm = HuggingFaceHub(
    repo_id="Harshgup16/llama-3-8b-Instruct-bnb-4bit-medical",
    huggingfacehub_api_token="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",  # Replace with your actual token
    task="text-generation",
    model_kwargs={
        "temperature": 0.7,
        "max_new_tokens": 500,
        "top_p": 0.9,
    }
)

# Create a prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the following medical question:
Question: {question}
Please provide a detailed and accurate response.
""")

# Create the chain
chain = LLMChain(llm=llm, prompt=prompt)

# Add routes to the FastAPI app
add_routes(
    app,
    chain,
    path="/medical",
)

# Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}