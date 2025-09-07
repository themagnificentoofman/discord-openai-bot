"""
Discord bot that uses a locally-hosted large language model to answer user questions via
a slash command.

This script registers a slash command called `/ask`. When invoked, it sends the user's
question to a locally served LLM (such as one running via Ollama) and returns the
assistant's response.

Environment variables (set via Heroku Config Vars or a .env file for local testing):
    DISCORD_TOKEN  - Bot token from the Discord Developer Portal
    LLM_BASE_URL   - Base URL of your local model server (e.g. https://my-tunnel.example.com)
    LLM_MODEL      - Name of the model to use (e.g. "llama3.1:8b")

To run locally, install the dependencies listed in requirements.txt, set the environment
variables (e.g. using a .env file and python-dotenv) and execute this script:

    pip install -r requirements.txt
    python bot.py

When deployed to Heroku, the Procfile will start this script as a worker process. The
model itself must be running elsewhere (on your machine or a GPU server) and
accessible via the LLM_BASE_URL.
"""

import os
import json
import aiohttp
import discord
from discord import app_commands


def get_env_var(name: str) -> str:
    """
    Retrieve an environment variable or raise an error if missing.

    Args:
        name: Name of the environment variable.

    Returns:
        The value of the environment variable.
    """
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value


async def call_local_llm(base_url: str, model: str, prompt: str) -> str:
    """
    Send a single-turn chat request to a locally hosted LLM via its REST API and
    return the assistant's text. This function expects an OpenAI-like chat
    interface such as Ollama's `/api/chat` endpoint.

    Args:
        base_url: Base URL of the local LLM server (without trailing slash).
        model: Name of the model to use on the server.
        prompt: User's question or message.

    Returns:
        The assistant's response as a string.

    Raises:
        RuntimeError: If the server returns a non-200 status code.
    """
    url = f"{base_url.rstrip('/')}/api/chat"
    payload: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"LLM server returned {resp.status}: {body}")
            data = await resp.json()
    # Expect response like {"message":{"role":"assistant","content":"..."}, ...}
    try:
        return data["message"]["content"].strip()
    except Exception:
        return json.dumps(data, indent=2)[:1900]


def create_client() -> tuple[discord.Client, app_commands.CommandTree]:
    """
    Instantiate the Discord client and command tree using credentials from
    environment variables. A local LLM server must be running and accessible
    via the LLM_BASE_URL.

    Returns:
        A tuple of (discord.Client, app_commands.CommandTree)
    """
    # Read mandatory environment variables
    discord_token = get_env_var("DISCORD_TOKEN")
    base_url = get_env_var("LLM_BASE_URL")
    model = get_env_var("LLM_MODEL")

    # Set up Discord client with minimal intents (slash commands only)
    intents = discord.Intents.default()
    client = discord.Client(intents=intents)
    tree = app_commands.CommandTree(client)

    # Register slash command
    @tree.command(name="ask", description="Ask the local model a question")
    @app_commands.describe(prompt="Your question for the model")
    async def ask(interaction: discord.Interaction, prompt: str) -> None:
        """Handle the /ask slash command."""
        # Indicate the bot is thinking while awaiting the model's response
        await interaction.response.defer(thinking=True)
        try:
            text = await call_local_llm(base_url, model, prompt)
        except Exception as e:
            text = f"Error communicating with local model: {e}"
        # Send truncated response to respect Discord message limits
        await interaction.followup.send(text[:1900] or "(no content)")

    @client.event
    async def on_ready() -> None:
        """Run when the bot is ready and sync slash commands."""
        await tree.sync()
        print(f"Logged in as {client.user}. Slash commands synced.")

    return client, tree


def main() -> None:
    """
    Entry point for running the bot. This function reads the necessary
    environment variables, instantiates the Discord client, and runs it.
    """
    client, _ = create_client()
    token = get_env_var("DISCORD_TOKEN")
    client.run(token)


if __name__ == "__main__":
    main()
