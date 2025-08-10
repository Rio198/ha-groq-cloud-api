"""Conversation support for Groq Cloud."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .entity import GroqConversationEntity


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    for sub_entry in config_entry.subentries.values():
        if sub_entry.subentry_type == "conversation":
            async_add_entities(
                [GroqConversationEntity(hass, config_entry, sub_entry)]
            )