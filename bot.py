"""
Discord bot that uses OpenAI's Responses API to answer user questions via a slash command.

This script uses discord.py to register a slash command called `/ask`. When invoked,
it calls the OpenAI Responses API with the provided prompt and sends back the reply.

Environment variables (set via Heroku Config Vars or a .env file for local testing):
    DISCORD_TOKEN   - Bot token from the Discord Developer Portal
    OPENAI_API_KEY  - OpenAI API key for accessing the Responses API
    OPENAI_MODEL    - (optional) Model name, defaults to gpt-4.1. Update to gpt-5 when available

To run locally, install the dependencies listed in requirements.txt, set the environment
variables (e.g. using a .env file and python-dotenv) and execute this script:

    pip install -r requirements.txt
    python bot.py

When deployed to Heroku, the Procfile will start this script as a worker process.
"""

import os
from openai import OpenAI
import discord
from discord import app_commands


def get_env_var(name: str) -> str:
    """Retrieve an environment variable or raise an error if missing."""
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value


def create_client() -> tuple[discord.Client, app_commands.CommandTree, OpenAI]:
    """
    Instantiate the Discord client, command tree, and OpenAI client using
    credentials from environment variables.

    Returns:
        A tuple of (discord.Client, app_commands.CommandTree, OpenAI)
    """
    discord_token = get_env_var("DISCORD_TOKEN")
    openai_api_key = get_env_var("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1")

    # Set up Discord client with minimal intents (no message content intent required for slash commands)
    intents = discord.Intents.default()
    client = discord.Client(intents=intents)
    tree = app_commands.CommandTree(client)

    # Initialize OpenAI client
    oai = OpenAI(api_key=openai_api_key)

    # Register slash command
    @tree.command(name="ask", description="Ask the AI a question (concise answers by default)")
    @app_commands.describe(prompt="Your question for the AI")
    async def ask(interaction: discord.Interaction, prompt: str) -> None:
        """Handle the /ask slash command."""
        # Show a temporary thinking state
        await interaction.response.defer(thinking=True)

        # Call the OpenAI Responses API
        try:
            response = oai.responses.create(
                model=model,
                input=prompt,
                instructions="You are a helpful Discord bot. Answer concisely unless asked to elaborate."
            )
        except Exception as e:
            await interaction.followup.send(f"Error communicating with OpenAI: {e}")
            return

        # Extract the text from the response. Prefer output_text if available.
        try:
            text = response.output_text  # type: ignore[attr-defined]
        except AttributeError:
            try:
                # Fallback: flatten the first text content if present
                text = response.output[0].content[0].text  # type: ignore[index]
            except Exception:
                text = "(no content)"

        # Respect Discord's message length limit (~2000 chars)
        await interaction.followup.send(text[:1900] or "(no content)")

    @client.event
    async def on_ready() -> None:
        """Run when the bot is ready and sync slash commands."""
        await tree.sync()
        print(f"Logged in as {client.user}. Slash commands synced.")

    return client, tree, oai


def main() -> None:
    """Entry point for running the bot."""
    client, _, _ = create_client()
    discord_token = get_env_var("DISCORD_TOKEN")
    client.run(discord_token)


if __name__ == "__main__":
    main()
