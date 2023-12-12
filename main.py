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

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:90.0) Gecko/20100101 Firefox/90.0'
}

#currently used LLM
LLM = ChatOpenAI(model="gpt-3.5-turbo-16k")

# duck duck go search api
DDG_SEARCH = DuckDuckGoSearchResults()

def choose_mode(mode):
    # create template from the promt and assign LLM for chaining
    prompt_template_concepts = "summarize the following content: {content}, use your tools to search and summarize content into a guide on how to use the software tools or library"
    prompt_template_library = "summarize the following content: {content}, use your tools to search and summarize content into an easily understandable explanations with examples of implementation"
    prompt_template_usecase  = "summarize the following content: {content}, use your tools to search and explain why this specific tool or library is the perfect tool for this use case" 
    
    if mode == "library":
        return prompt_template_library
    else if mode == "usecase":
        return prompt_template_usecase
    else:
        return prompt_template_concepts
    
#parsing html for returning web page 
def parse_html(content) -> str:
    soup = BeautifulSoup(content, 'html.parser')
    text_content_with_links = soup.get_text()
    return text_content_with_links

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
def web_tool_definition(fetch_web_page): 
    return Tool.from_function(
        func=fetch_web_page,
        name="WebFetcher",
        description="Fetches the content of a web page"
    )

def llm_chain_definition(llm, prompt_template):
    #prompt template for defining input template
    return LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(prompt_template)
    )

def summarize_tool_definition(llm_chain_used):
    # setting up summarizer 
    return Tool.from_function(
        func=llm_chain_used.run,
        name="Summarizer",
        description="Summarizes a web page"
    )

def initialize_agent_definition(tools, llm):
    # initiate agent for executing the prompt
    return initialize_agent(
        tools=tools,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        llm=llm,
        verbose=True
    )


@click.command()
@click.argument('mode', type=str, nargs=-1)
def maincommand(mode):
    click.echo("==============================================")
    click.echo("===== Welcome to Autonomous Agent Mikiya =====")
    click.echo("==============================================")
    click.echo("You can choose between these modes as the arguments:")
    click.echo("concepts : explain to you the concepts of tools or library with a summary") 
    click.echo("library : explain to you how tools or library works and the implementation")
    click.echo("usecase : explain to you what is this tool used for and their use case")

    prompt_template_used = choose_mode(mode)

    web_fetch_tool = web_tool_definition(fetch_web_page)
    llm_chain = llm_chain_definition(LLM, prompt_template_used)
    summarize_tool = summarize_tool_definition(llm_chain)
    tools = [DDG_SEARCH, web_fetch_tool, summarize_tool]
    agent = initialize_agent_definition(tools, LLM)

    #agent type zero shot react will keep questioning themself when executing the chain
    user_input = input("what would you like to learn today? ")
    #example "Research how to use the requests library in python, use your tools to search and summarize content into a guide on how to use the requests library"
    print(agent.run(user_input))

if __name__ == '__main__':
    maincommand()