from enum import Enum
from typing import Dict, List, Optional, Union

from haystack import Document, component, logging
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator



logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """
    Currently LLM providers supported by `LLMMetadataExtractor`.
    """

    OPENAI = "openai"

    @staticmethod
    def from_str(string: str) -> "LLMProvider":
        """
        Convert a string to a LLMProvider enum.
        """
        provider_map = {e.value: e for e in LLMProvider}
        provider = provider_map.get(string)
        if provider is None:
            msg = (
                f"Invalid LLMProvider '{string}'"
                f"Supported LLMProviders are: {list(provider_map.keys())}"
            )
            raise ValueError(msg)
        return provider


@component
class LLMRanker:
    """
    """
    def __init__(self, query: str, docs: List[Document], prompt: PromptBuilder, llm_generator: LLMProvider = Union[str, LLMProvider], top_k: Optional[int] = 10):
        """
        """
        template = """
                    For the list of given documents:
                        {% for doc in documents %}
                            {{ doc.content }}
                        {% endfor %}

                    rank these documents based on the given {{query}}
                    and then return the {{top_k}} or all the documents which ever is minimum.
                   """
        self.query = query
        self.docs = docs
        self.prompt = prompt or PromptBuilder(template=template, required_variables=["query", "documents"])
        self.llm_genertor = llm_generator
        self.top_k = top_k

        @component.output_types(documents=List[Document])
        def run(self, ):
            pass
