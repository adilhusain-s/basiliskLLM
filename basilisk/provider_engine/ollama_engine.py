import logging
from functools import cached_property
from typing import TYPE_CHECKING, Union, Generator

import asyncio
from ollama import AsyncClient, ProcessResponse, chat

from basilisk.conversation import (
	Conversation,
	Message,
	MessageBlock,
	MessageRoleEnum,
)
from basilisk.provider_ai_model import ProviderAIModel
from basilisk.provider_capability import ProviderCapability
from .base_engine import BaseEngine

log = logging.getLogger(__name__)


class OllamaEngine(BaseEngine):
	capabilities: set[ProviderCapability] = {
		ProviderCapability.TEXT  # Only TEXT capability since TTS and STT are not implemented
	}

	def __init__(self, account: Account) -> None:
		super().__init__(account)
		self.client = AsyncClient()  # Initialize the async client

	@cached_property
	def client(self):
		return self.client  # Should be a cached property

	@cached_property
	def models(self) -> list[ProviderAIModel]:
		"""
		Get models dynamically from Ollama.
		"""
		log.debug("Fetching available Ollama models...")

		# Fetching the list of models from the Ollama SDK synchronously
		response: ProcessResponse = self.client.ps()
		models = []

		for model in response.models:
			models.append(
				ProviderAIModel(
					id=model.model,
					description=model.details,
					context_window=model.size_vram,  # Example; adjust this as needed
					max_output_tokens=model.size,  # Example; adjust as needed
					max_temperature=2.0,  # Example; adjust as needed
				)
			)

		return models

	def completion(
		self,
		new_block: MessageBlock,
		conversation: Conversation,
		system_message: Message | None,
		**kwargs,
	) -> Union[dict, MessageBlock]:
		"""
		Handle Ollama chat completions.
		"""
		params = {
			'model': new_block.model.id,
			'messages': self.get_messages(
				new_block, conversation, system_message
			),
		}

		# Run the async function synchronously using asyncio.run
		loop = asyncio.get_event_loop()
		response = loop.run_until_complete(self._async_completion(**params))

		# Adjust this to match your expected response format
		return MessageBlock(response)

	async def _async_completion(self, **params):
		"""
		Async helper for the completion method.
		"""
		response = await chat(**params)
		return response

	def completion_response_with_stream(
		self, stream: Generator[dict, None, None], **kwargs
	):
		"""
		Handle streaming responses asynchronously.
		"""
		loop = asyncio.get_event_loop()
		return loop.run_until_complete(
			self._async_completion_response_with_stream(stream, **kwargs)
		)

	async def _async_completion_response_with_stream(
		self, stream: Generator[dict, None, None], **kwargs
	):
		"""
		Async helper for handling streaming responses.
		"""
		async for chunk in stream:
			content = chunk.get('message', {}).get('content', '')
			if content:
				yield self.normalize_linesep(content)

	def completion_response_without_stream(
		self, response: dict, new_block: MessageBlock, **kwargs
	) -> MessageBlock:
		"""
		Handle non-streaming responses.
		"""
		new_block.response = Message(
			role=MessageRoleEnum.ASSISTANT,
			content=self.normalize_linesep(
				response.get('message', {}).get('content', '')
			),
		)
		return new_block
