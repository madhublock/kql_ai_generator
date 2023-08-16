# using the index created above we are connecting to llm using langchain and asking question
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

openai_api_key = os.environ["OPENAI_API_KEY"]
start_time = time.time()
urls = [
    "https://learn.microsoft.com/en-us/azure/azure-monitor/app/data-model-complete"
]

loader = UnstructuredURLLoader(urls=urls)
data = loader.load()
print(f"loaded {len(data)} documents")
docs_load_time = time.time()
print("--- Docs Load time  %s seconds ---" % (docs_load_time - start_time))
# for file in files:
#     loader = BSHTMLLoader(file)
#     data = loader.load()
#     docs.append(data)
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
)
documents = text_splitter.split_documents(data)
print(f"Splitting into {len(documents)} chunks")
split_load_time = time.time()
print(
    "--- Splitting Load time  %s seconds ---" % (split_load_time - docs_load_time)
)
embeddings = OpenAIEmbeddings(
    deployment="ada-embeddings", chunk_size="1", model="text-embedding-ada-002"
)
vectorstore = FAISS.from_documents(documents, embeddings)
faiss_load_time = time.time()
print("--- FAISS Load time  %s seconds ---" % (faiss_load_time - split_load_time))
# db.save_local("kqltool_multi_index")
faiss_save_time = time.time()
print("--- FAISS saving time  %s seconds ---" % (faiss_save_time - faiss_load_time))
print("--- Total time  %s seconds ---" % (faiss_save_time - start_time))

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
'''nextVersionPrompt: You understand that the user's request may not always be precise or may require additional context. In such cases, you are capable of asking clarifying questions to ensure that you fully understand the user's needs before generating the Kusto query.
'''
prompt_template = """

    You are an AI assistant with expertise in Azure Application Insights and Kusto Query Language (KQL). You have access to all the application insights within the BlockWorks Online Azure DevOps project. Your goal is to assist users in retrieving the data they need by generating the appropriate Kusto queries based on their requests.

    You are also aware that the data you have access to is extensive and may contain many different tables and fields. Therefore, you are careful to generate queries that are as specific as possible to the user's request, to avoid returning unnecessary or irrelevant data.

    If you don't know the answer, just say that you don't know, attempt to ask clarifying questions still to the user, but do not 
    hallucinate and make up a Kusto query that is wrong or does not exist. 
    
    Here is the user's request: {request}

    Based on this request, please generate the Kusto query that will retrieve the requested data. If you need more information to generate the query, please specify what additional information you need.
                """




qa_prompt = PromptTemplate(
    template=prompt_template, input_variables=["request"]
)
llm = AzureOpenAI(
        deployment_name="dv3llm01-01",
        model_name="text-embedding-ada-002",
        verbose=True,
        temperature=0,
)

doc_chain = load_qa_chain(
        llm, chain_type="stuff", prompt=qa_prompt
)
qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), memory=memory)


# qa = ConversationalRetrievalQAChain.fromLLM(

# )
# qa = ConversationalRetrievalChain(
#         combine_docs_chain=doc_chain,
#         retriever=db.as_retriever(),
#         return_source_documents=True
# )

#qt = """get all requests which contains end point of "api" """

print(prompts.response.get('answer'))