from pyexpat import model
from urllib import response
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import Tool, DuckDuckGoSearchResults
from langchain.prompts import PromptTemplate #template untuk memperlengkap/jelas deskripsi prompt user (legacy)
from langchain.chat_models import ChatOpenAI #model AI yang dipakai
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType
import click # for cli command prettier

# for loading env variables
load_dotenv()

# duck duck go search api
ddg_search = DuckDuckGoSearchResults()

#parsing html for returning web page 
def parse_html(content) -> str:
    soup = BeautifulSoup(content, 'html.parser')
    text_content_with_links = soup.get_text()
    return text_content_with_links

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:90.0) Gecko/20100101 Firefox/90.0'
}

#operation for fetching the url
def fetch_web_page(url: str) -> str:
    try:
        response = requests.get(url, headers=HEADERS)
        # response content in raw binary state like images
        print("response content: ", response.content)
        return parse_html(response.content)
    except Exception as e:
        return f"error, failed to get the url: {e}"

#fetch web using this tool 
web_fetch_tool = Tool.from_function(
    func=fetch_web_page,
    name="WebFetcher",
    description="Fetches the content of a web page"
)

# create template from the promt and assign LLM for chaining
prompt_template_concepts = "summarize the following content: {content}, use your tools to search and summarize content into a guide on how to use the requests library"
prompt_template_library = "summarize the following content: {content}, use your tools to search and summarize content into an easily understandable explanations with examples of implementation"
#currently used LLM
llm = ChatOpenAI(model="gpt-3.5-turbo-16k")

#prompt template for defining input template
llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template_concepts)
)

# setting up summarizer 
summarize_tool = Tool.from_function(
    func=llm_chain.run,
    name="Summarizer",
    description="Summarizes a web page"
)

# initializing the tools from function to use by the agent
tools = [ddg_search, web_fetch_tool, summarize_tool]

# initiate agent for executing the prompt
agent = initialize_agent(
    tools=tools,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm=llm,
    verbose=True
)

@click.command()
def maincommand():
    click.echo("==============================================")
    click.echo("===== Welcome to Autonomous Agent Mikiya =====")
    click.echo("==============================================")

    #agent type zero shot react will keep questioning themself when executing the chain
    user_input = input("what would you like to learn today? ")
    #example "Research how to use the requests library in python, use your tools to search and summarize content into a guide on how to use the requests library"
    print(agent.run(user_input))

if __name__ == '__main__':
    maincommand()