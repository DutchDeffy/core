"""The Anthropic Claude AI Conversation integration."""

from __future__ import annotations

# from typing import Literal, cast
import anthropic

# import voluptuous as vol
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import HomeAssistant

# ServiceCall,
# ServiceResponse,
# SupportsResponse,
from homeassistant.exceptions import ConfigEntryNotReady

# HomeAssistantError,
# ServiceValidationError,
from homeassistant.helpers import config_validation as cv

# issue_registry as ir,
# selector,
from homeassistant.helpers.typing import ConfigType

from .const import DOMAIN, LOGGER

PLATFORMS = (Platform.CONVERSATION,)
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

type AnthropicConfigEntry = ConfigEntry[anthropic.AsyncClient]


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up Anthtropic Claude AI Conversation."""
    return True


async def async_setup_entry(hass: HomeAssistant, entry: AnthropicConfigEntry) -> bool:
    """Set up Anthtropic Claude AI Conversation from a config entry."""
    client = anthropic.Asyncanthropic(api_key=entry.data[CONF_API_KEY])
    try:
        await hass.async_add_executor_job(client.with_options(timeout=10.0).models.list)
    except anthropic.AuthenticationError as err:
        LOGGER.error("Invalid API key: %s", err)
        return False
    except anthropic.AnthropicError as err:
        raise ConfigEntryNotReady(err) from err

    entry.runtime_data = client

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload Anthtropic Claude AI."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
