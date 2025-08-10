"""AI Task support for Groq Cloud."""

from __future__ import annotations

import groq

from homeassistant.components import ai_task
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up AI task entities."""
    for sub_entry in config_entry.subentries.values():
        if sub_entry.subentry_type == "ai_task_data":
            async_add_entities(
                [AITask(hass, config_entry, sub_entry)]
            )


class AITask(ai_task.AITask):
    """AI Task for Groq Cloud."""

    def __init__(
        self, hass: HomeAssistant, entry: ConfigEntry, sub_entry: ConfigSubentry
    ) -> None:
        """Initialize the AI task."""
        self.hass = hass
        self.entry = entry
        self.sub_entry = sub_entry
        self._attr_name = sub_entry.title
        self._attr_unique_id = sub_entry.subentry_id

    async def async_execute(
        self,
        task_input: ai_task.TaskInput,
    ) -> ai_task.TaskOutput:
        """Execute the task."""
        options = self.sub_entry.data
        client: groq.AsyncClient = self.entry.runtime_data

        messages = [
            {"role": "user", "content": task_input.prompt}
        ]

        try:
            response = await client.chat.completions.create(
                model=options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
                messages=messages,
                max_tokens=options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
                top_p=options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                temperature=options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
            )
        except groq.GroqError as err:
            raise HomeAssistantError(f"Error executing AI task: {err}") from err

        return ai_task.TaskOutput(
            response.choices[0].message.content
        )
