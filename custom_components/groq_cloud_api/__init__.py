"""The Groq Cloud API integration."""

from __future__ import annotations

import groq

import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.exceptions import (
    ConfigEntryNotReady,
    HomeAssistantError,
    ServiceValidationError,
)
from homeassistant.helpers import config_validation as cv, selector
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.typing import ConfigType

from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DOMAIN,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)

SERVICE_GENERATE_CONTENT = "generate_content"

PLATFORMS = (Platform.AI_TASK, Platform.CONVERSATION)
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

type GroqConfigEntry = ConfigEntry[groq.AsyncClient]


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up Groq Cloud API."""

    async def send_prompt(call: ServiceCall) -> ServiceResponse:
        """Send a prompt to Groq and return the response."""
        entry_id = call.data["config_entry"]
        entry = hass.config_entries.async_get_entry(entry_id)

        if entry is None or entry.domain != DOMAIN:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="invalid_config_entry",
                translation_placeholders={"config_entry": entry_id},
            )

        # Get first conversation subentry for options
        conversation_subentry = next(
            (
                sub
                for sub in entry.subentries.values()
                if sub.subentry_type == "conversation"
            ),
            None,
        )
        if not conversation_subentry:
            raise ServiceValidationError("No conversation configuration found")

        model: str = conversation_subentry.data.get(
            CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL
        )
        client: groq.AsyncClient = entry.runtime_data

        messages = [
            {"role": "user", "content": call.data[CONF_PROMPT]}
        ]

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=conversation_subentry.data.get(
                    CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS
                ),
                top_p=conversation_subentry.data.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                temperature=conversation_subentry.data.get(
                    CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE
                ),
                user=call.context.user_id,
            )

        except groq.GroqError as err:
            raise HomeAssistantError(f"Error generating content: {err}") from err

        return {"text": response.choices[0].message.content}

    hass.services.async_register(
        DOMAIN,
        SERVICE_GENERATE_CONTENT,
        send_prompt,
        schema=vol.Schema(
            {
                vol.Required("config_entry"): selector.ConfigEntrySelector(
                    {
                        "integration": DOMAIN,
                    }
                ),
                vol.Required(CONF_PROMPT): cv.string,
            }
        ),
        supports_response=SupportsResponse.ONLY,
    )

    return True


async def async_setup_entry(hass: HomeAssistant, entry: GroqConfigEntry) -> bool:
    """Set up Groq Cloud API from a config entry."""
    client = groq.AsyncGroq(
        api_key=entry.data[CONF_API_KEY],
        http_client=get_async_client(hass),
    )

    try:
        await hass.async_add_executor_job(client.with_options(timeout=10.0).models.list)
    except groq.AuthenticationError as err:
        LOGGER.error("Invalid API key: %s", err)
        return False
    except groq.GroqError as err:
        raise ConfigEntryNotReady(err) from err

    entry.runtime_data = client

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    entry.async_on_unload(entry.add_update_listener(async_update_options))

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload Groq Cloud API."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_update_options(hass: HomeAssistant, entry: GroqConfigEntry) -> None:
    """Update options."""
    await hass.config_entries.async_reload(entry.entry_id)
