from fastapi import FastAPI, HTTPException, WebSocket
from pydantic import BaseModel
import logging
import autogen
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from autogen.io.websockets import IOWebsockets
from autogen.agentchat import AssistantAgent
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
from fastapi.middleware.cors import CORSMiddleware
import json
import requests
from typing import Union, Iterable, Optional
from websockets.exceptions import ConnectionClosed

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    prompt: str

setup_logging()
logger = logging.getLogger(__name__)

def configure_agent(assistant_type: str, stream: bool = False) -> GPTAssistantAgent:
    try:
        logger.info("Configuring GPT Assistant Agent...")
        assistant_id = load_assistant_id(assistant_type)
        llm_config = GetConfig().config_list
        oai_config = {
            "config_list": llm_config["config_list"], "assistant_id": assistant_id}
        if stream:
            oai_config["stream"] = True
        gpt_assistant = GPTAssistantAgent(
            name=assistant_type, instructions=AssistantAgent.DEFAULT_SYSTEM_MESSAGE, llm_config=oai_config
        )
        logger.info("GPT Assistant Agent configured.")
        return gpt_assistant
    except openai.NotFoundError:
        logger.warning("Assistant not found. Creating new assistant...")
        create_agent(assistant_type)
        return configure_agent(assistant_type, stream)
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
        user_proxy = create_user_proxy()
        response = user_proxy.initiate_chat(gpt_assistant, message=prompt_request.prompt)
        return {"response": response}
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")   
        raise HTTPException(status_code=500, detail="Internal Server Error")

class FastAPIServerConnection:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    async def send(self, message: Union[str, bytes, Iterable[Union[str, bytes]]]) -> None:
        if isinstance(message, (str, bytes)):
            await self.websocket.send_text(message if isinstance(message, str) else message.decode('utf-8'))
        else:
            for part in message:
                await self.send(part)

    async def recv(self, timeout: Optional[float] = None) -> str:
        try:
            return await self.websocket.receive_text()
        except ConnectionClosed as e:
            logger.error(f"Connection closed: {e}")
            raise e
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            raise e

    async def close(self) -> None:
        await self.websocket.close()

async def on_connect(iostream: IOWebsockets) -> None:
    print(f" - on_connect(): Connected to client using IOWebsockets {iostream}", flush=True)
    print(" - on_connect(): Receiving message from client.", flush=True)

    try:
        # Receive Initial Message
        initial_msg = await iostream.input()
        print(f"Received initial message: {initial_msg}", flush=True)

        # Instantiate GPT Assistant Agent with streaming
        gpt_assistant = configure_agent("BrowsingAgent", stream=True)
        register_functions(gpt_assistant)
        user_proxy = create_user_proxy()

        # Initiate conversation
        print(
            f" - on_connect(): Initiating chat with agent {gpt_assistant} using message '{initial_msg}'",
            flush=True,
        )
        response = user_proxy.initiate_chat(gpt_assistant, message=initial_msg)
        await iostream.output(response)
    except Exception as e:
        print(f"Exception in on_connect: {e}", flush=True)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    server_conn = FastAPIServerConnection(websocket)
    iostream = IOWebsockets(server_conn)
    await on_connect(iostream)

if __name__ == "__main__":
    import uvicorn
    print("Running app on port 3030")
    uvicorn.run(app, host="0.0.0.0", port=3030)
