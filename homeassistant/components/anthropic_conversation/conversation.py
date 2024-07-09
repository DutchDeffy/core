"""Conversation support for Anthropic Claude AI."""

from typing import Literal

import anthropic
from anthropic._types import NOT_GIVEN
from anthropic.types import MessageParam, ToolParam

# ChatCompletionUserMessageParam,
from homeassistant.components import assist_pipeline, conversation
from homeassistant.components.conversation import trace
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError, TemplateError
from homeassistant.helpers import device_registry as dr, intent, llm, template
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util import ulid

from . import AnthropicConfigEntry
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

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: AnthropicConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    agent = AnthropicConversationEntity(config_entry)
    async_add_entities([agent])


# TO do FORMAT tool


def _format_tool(tool: llm.Tool) -> ToolParam:
    tool_name = 0
    return ToolParam(input_schema=Literal["object"], name=tool_name)


class AnthropicConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """Anthropic conversation agent."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: AnthropicConfigEntry) -> None:
        """Initialize the agent."""
        self.entry = entry
        self.history: dict[str, list[MessageParam]] = {}
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="Anthropic",
            model="Claude",
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        assist_pipeline.async_migrate_engine(
            self.hass, "conversation", self.entry.entry_id, self.entity_id
        )
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process a sentence."""
        options = self.entry.options
        intent_response = intent.IntentResponse(language=user_input.language)
        llm_api: llm.APIInstance | None = None
        tools: list[ToolParam] | None = None
        user_name: str | None = None
        llm_context = llm.LLMContext(
            platform=DOMAIN,
            context=user_input.context,
            user_prompt=user_input.text,
            language=user_input.language,
            assistant=conversation.DOMAIN,
            device_id=user_input.device_id,
        )

        if options.get(CONF_LLM_HASS_API):
            try:
                llm_api = await llm.async_get_api(
                    self.hass,
                    options[CONF_LLM_HASS_API],
                    llm_context,
                )
            except HomeAssistantError as err:
                LOGGER.error("Error getting LLM API: %s", err)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    f"Error preparing LLM API: {err}",
                )
                return conversation.ConversationResult(
                    response=intent_response, conversation_id=user_input.conversation_id
                )
            tools = [_format_tool(tool) for tool in llm_api.tools]

        if user_input.conversation_id is None:
            conversation_id = ulid.ulid_now()
            messages = []

        elif user_input.conversation_id in self.history:
            conversation_id = user_input.conversation_id
            messages = self.history[conversation_id]

        else:
            # Conversation IDs are ULIDs. We generate a new one if not provided.
            # If an old OLID is passed in, we will generate a new one to indicate
            # a new conversation was started. If the user picks their own, they
            # want to track a conversation and we respect it.
            try:
                ulid.ulid_to_bytes(user_input.conversation_id)
                conversation_id = ulid.ulid_now()
            except ValueError:
                conversation_id = user_input.conversation_id

            messages = []

        if (
            user_input.context
            and user_input.context.user_id
            and (
                user := await self.hass.auth.async_get_user(user_input.context.user_id)
            )
        ):
            user_name = user.name

        try:
            prompt_parts = [
                template.Template(
                    llm.BASE_PROMPT
                    + options.get(CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT),
                    self.hass,
                ).async_render(
                    {
                        "ha_name": self.hass.config.location_name,
                        "user_name": user_name,
                        "llm_context": llm_context,
                    },
                    parse_result=False,
                )
            ]

        except TemplateError as err:
            LOGGER.error("Error rendering prompt: %s", err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Sorry, I had a problem with my template: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        if llm_api:
            prompt_parts.append(llm_api.api_prompt)

        prompt = "\n".join(prompt_parts)

        # Create a copy of the variable because we attach it to the trace
        messages = [{"role": "user", "content": {user_input.text}}]

        LOGGER.debug("System Prompt: %s\t User Prompt: %s", prompt, messages)
        trace.async_conversation_trace_append(
            trace.ConversationTraceEventType.AGENT_DETAIL, {"messages": messages}
        )

        client = self.entry.runtime_data

        # To prevent infinite loops, we limit the number of iterations
        for _iteration in range(MAX_TOOL_ITERATIONS):
            try:
                response = await client.messages.create(
                    model=options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
                    system=prompt,
                    messages=messages,
                    tools=tools or NOT_GIVEN,
                    max_tokens=options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS),
                    top_p=options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                    temperature=options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
                    user=conversation_id,
                )
            except anthropic.AnthropicError as err:
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    f"Sorry, I had a problem talking to Anthropic: {err}",
                )
                return conversation.ConversationResult(
                    response=intent_response, conversation_id=conversation_id
                )

            LOGGER.debug("Response %s", response)
            # !To be done convert message ()

            messages.append(response)
            # tools = response.tool_use

        self.history[conversation_id] = messages

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(response.content or "")
        return conversation.ConversationResult(
            response=intent_response, conversation_id=conversation_id
        )
