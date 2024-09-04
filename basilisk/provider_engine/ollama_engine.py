import logging
from functools import cached_property
from typing import TYPE_CHECKING

import ollama

from basilisk.conversation import (
	Conversation,
	Message,
	MessageBlock,
	MessageRoleEnum,
)

from .base_engine import BaseEngine, ProviderAIModel, ProviderCapability

if TYPE_CHECKING:
	pass

logger = logging.getLogger(__name__)


class OllamaEngine(BaseEngine):
	capabilities: set[ProviderCapability] = {
		ProviderCapability.TEXT,
		ProviderCapability.IMAGE,
	}

	def __init__(self, account) -> None:
		super().__init__(account)

	@property
	def client(self) -> None:
		"""
		Property to return the client object
		"""
		raise NotImplementedError("Getting client not supported for Ollama")

	@cached_property
	def models(self) -> list[ProviderAIModel]:
		"""
		Get models
		"""
		# TODO: should be fetched from user account
		return [
			ProviderAIModel(
				id="llama3.1",
				context_window=1048576,
				max_output_tokens=8192,
				vision=False,
				default_temperature=1.0,
			)
		]

	def completion(
		self,
		new_block: MessageBlock,
		conversation: Conversation,
		system_message: Message | None,
		**kwargs,
	):
		super().completion(new_block, conversation, system_message, **kwargs)
		params = {
			"model": new_block.model.id,
			"messages": self.get_messages(
				new_block, conversation, system_message
			),
			# "temperature": new_block.temperature,
			# "top_p": new_block.top_p,
			"stream": new_block.stream,
		}
		if new_block.max_tokens:
			params["max_tokens"] = new_block.max_tokens
		params.update(kwargs)
		response = ollama.chat(**params)
		return response

	def completion_response_with_stream(self, stream):
		for chunk in stream:
			content = chunk["message"]["content"]
			if content:
				yield self.normalize_linesep(content)

	def completion_response_without_stream(self, response, new_block, **kwargs):
		new_block.response = Message(
			role=MessageRoleEnum.ASSISTANT,
			content=self.normalize_linesep(response["message"]["content"]),
		)
		return new_block
