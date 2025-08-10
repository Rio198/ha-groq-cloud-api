"""Constants for the Groq Cloud API integration."""

import logging
from types import MappingProxyType

from homeassistant.helpers import llm

DOMAIN = "groq_cloud_api"
LOGGER = logging.getLogger(__name__)

DEFAULT_NAME = "Groq Cloud API"
DEFAULT_CONVERSATION_NAME = "Groq Conversation"
DEFAULT_AI_TASK_NAME = "Groq AI Task"

CONF_RECOMMENDED = "recommended"
CONF_PROMPT = "prompt"
CONF_CHAT_MODEL = "chat_model"
RECOMMENDED_CHAT_MODEL = "qwen/qwen3-32b"
CONF_MAX_TOKENS = "max_tokens"
RECOMMENDED_MAX_TOKENS = 150
CONF_TOP_P = "top_p"
RECOMMENDED_TOP_P = 0.95
CONF_TEMPERATURE = "temperature"
RECOMMENDED_TEMPERATURE = 0.6

RECOMMENDED_CONVERSATION_OPTIONS = MappingProxyType(
    {
        CONF_RECOMMENDED: True,
        CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
    }
)

RECOMMENDED_AI_TASK_OPTIONS = MappingProxyType(
    {
        CONF_RECOMMENDED: True,
    }
)