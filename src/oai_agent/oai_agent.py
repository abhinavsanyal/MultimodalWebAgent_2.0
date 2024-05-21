from fastapi import FastAPI, HTTPException, WebSocket
from pydantic import BaseModel
import logging
import autogen
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from src.configs.logging.logging_config import setup_logging
from src.oai_agent.utils.load_assistant_id import load_assistant_id
from src.oai_agent.utils.create_oai_agent import create_agent
from src.autogen_configuration.autogen_config import GetConfig
from src.tools.read_url import read_url
from src.tools.scroll import scroll
from src.tools.jump_to_search_engine import jump_to_search_engine
from src.tools.go_back import go_back
from src.tools.wait import wait
from src.tools.click_element import click_element
from src.tools.input_text import input_text
from src.tools.analyze_content import analyze_content
from src.tools.save_to_file import save_to_file

import openai
from autogen.agentchat import AssistantAgent
from fastapi.middleware.cors import CORSMiddleware

import websockets
import json
import requests

from src.webdriver.webdriver import WebDriver
from src.tools.utils.get_webdriver_instance import get_webdriver_instance

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class PromptRequest(BaseModel):
    prompt: str

setup_logging()
logger = logging.getLogger(__name__)

def configure_agent(assistant_type: str) -> GPTAssistantAgent:
    try:
        logger.info("Configuring GPT Assistant Agent...")
        assistant_id = load_assistant_id(assistant_type)
        llm_config = GetConfig().config_list
        oai_config = {
            "config_list": llm_config["config_list"], "assistant_id": assistant_id}
        gpt_assistant = GPTAssistantAgent(
            name=assistant_type, instructions=AssistantAgent.DEFAULT_SYSTEM_MESSAGE, llm_config=oai_config
        )
        logger.info("GPT Assistant Agent configured.")
        return gpt_assistant
    except openai.NotFoundError:
        logger.warning("Assistant not found. Creating new assistant...")
        create_agent(assistant_type)
        return configure_agent()
    except Exception as e:
        logger.error(f"Unexpected error during agent configuration: {str(e)}")
        raise

def register_functions(agent):
    logger.info("Registering functions...")
    function_map = {
        "analyze_content": analyze_content,
        "click_element": click_element,
        "go_back": go_back,
        "input_text": input_text,
        "jump_to_search_engine": jump_to_search_engine,
        "read_url": read_url,
        "scroll": scroll,
        "wait": wait,
        "save_to_file": save_to_file,
    }
    agent.register_function(function_map=function_map)
    logger.info("Functions registered.")

def create_user_proxy():
    logger.info("Creating User Proxy Agent...")
    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
        human_input_mode="NEVER",
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False,
        },
    )
    logger.info("User Proxy Agent created.")
    return user_proxy


@app.post("/get-web-agent-response")
def get_response(prompt_request: PromptRequest):
    try:
        gpt_assistant = configure_agent("BrowsingAgent")
        register_functions(gpt_assistant)
        user_proxy = create_user_proxy();
        response = user_proxy.initiate_chat(gpt_assistant, message=prompt_request.prompt)
        # get the browser instance and close the browser once done
        # WebDriver.getInstance().closeDriver();
        return {"response": response}
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")   
        # WebDriver.getInstance().closeDriver();
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    print("Running app on port 3000")
    uvicorn.run(app, host="0.0.0.0", port=3030)
