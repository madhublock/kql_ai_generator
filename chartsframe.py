from langchain.prompts.prompt import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import AzureOpenAI
import prompts
import os
import time
from os import getcwd, listdir
from os.path import isfile, join

from langchain.document_loaders import BSHTMLLoader, DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import AzureOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2022-12-01"
os.environ["OPENAI_API_BASE"] = "https://dv3llm01.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "53e94f3c907a4f9e8042af2494073ed0"


def langchain_charts(query, ans, df):
    llm = AzureOpenAI(
        deployment_name="dv3llm01-01",
        model_name="text-embedding-ada-002",
        verbose=True,
        temperature=0,
    )

    prompt_template = """

        You are an AI assistant with expertise in Pandas dataframe, Azure Application Insights and Kusto Query Language (KQL).
        You have access to all the application insights within the BlockWorks Online Azure DevOps project. 
        Your goal is to get the required columns for data frame to draw a chart on 2d axis (x and y columns) based on the KQL query request below.
        Also give the chart type and chart title based on the user's request.
         
        and chart type with required columns to draw a chart based on the user's request.

        You are passed the KQL QUERY: {request}
        and the sample pandas dataframe of 2 rows: {df}
        
        
        Please pass the response like below example:
         example:- Answer: xcol: column1, ycol:column2, chartType: bar , chartTitle: title1
        
    
        If you don't know the answer, just reply chart type as unknown and reason.
        example: Answer: chartType: unknown, reason: The reason I can't answer this question is because I can't find columns in the data for 2d chart.
        but do not hallucinate and make up a Kusto query that is wrong or does not exist. 
    If you do, you will be penalized. """

    # Fill in the user's request in the prompt template
    filled_prompt = prompt_template.format(request=ans, df=df)

    # Use the Langchain model to generate a response
    response = llm.generate([filled_prompt])  # Note the list around filled_prompt

    # Extract the generated text from the response
    generated_text = response.generations[0][0].text

    # Return the generated text
    return generated_text

