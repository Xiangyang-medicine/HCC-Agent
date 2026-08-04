"""
Configuration for the HCC Prognosis Assessment Multi-Agent System.
"""

from pydantic import BaseModel, Field
from typing import Optional
import os


class APIConfig(BaseModel):
    """API configuration for LLM backend."""
    # Claude API settings (via Chinese proxy)
    api_base: str = Field(
        default="https://rsxermu666.cn/v1",
        description="API base URL for the proxy"
    )
    api_key: str = Field(
        default="sk-308KwjH0x1DHdKBm9hxJf25bYWWPdrMI1JXTfvKew5Ki1ERC",
        description="API key for authentication"
    )
    model_name: str = Field(
        default="claude-opus-4-8",
        description="Model name to use"
    )
    max_tokens: int = Field(
        default=4096,
        description="Maximum tokens in response"
    )
    temperature: float = Field(
        default=0.7,
        description="Temperature for response generation"
    )
    timeout: int = Field(
        default=120,
        description="Request timeout in seconds"
    )


class DataConfig(BaseModel):
    """Data directory configuration."""
    data_dir: str = Field(
        default="F:/ACM/data",
        description="Directory for data files"
    )
    tcga_data_file: str = Field(
        default="tcga_lihc_data.parquet",
        description="TCGA-LIHC dataset file"
    )
    kegg_cache_dir: str = Field(
        default="F:/ACM/data/kegg_cache",
        description="KEGG pathway cache directory"
    )
    literature_cache_dir: str = Field(
        default="F:/ACM/data/literature_cache",
        description="Literature cache directory"
    )


class AgentConfig(BaseModel):
    """Agent behavior configuration."""
    # Coordinator settings
    coordinator_model: str = Field(
        default="claude-opus-4-8",
        description="Model for coordinator agent"
    )
    # Feature extraction settings
    feature_extraction_model: str = Field(
        default="claude-opus-4-8",
        description="Model for feature extraction"
    )
    # Literature agent settings
    literature_model: str = Field(
        default="claude-opus-4-8",
        description="Model for literature agent"
    )
    max_literature_results: int = Field(
        default=10,
        description="Maximum literature search results"
    )
    # Reasoning agent settings
    reasoning_model: str = Field(
        default="claude-opus-4-8",
        description="Model for reasoning agent"
    )


class Config:
    """Global configuration container."""

    def __init__(self):
        # Load environment variables for API key
        self.api = APIConfig()
        self.data = DataConfig()
        self.agent = AgentConfig()

        # Override from environment variables if set
        self.api.api_key = os.environ.get("CLAUDE_API_KEY", self.api.api_key)
        self.api.api_base = os.environ.get("CLAUDE_API_BASE", self.api.api_base)

    def set_api_key(self, api_key: str):
        """Set API key programmatically."""
        self.api.api_key = api_key

    def validate(self) -> bool:
        """Validate configuration."""
        if not self.api.api_key:
            print("Warning: API key not set. Please set CLAUDE_API_KEY environment variable.")
            return False
        return True


# Global config instance
config = Config()


def load_config(api_key: Optional[str] = None, api_base: Optional[str] = None) -> Config:
    """
    Load and configure the application.

    Args:
        api_key: Optional API key override
        api_base: Optional API base URL override

    Returns:
        Configured Config instance
    """
    if api_key:
        config.set_api_key(api_key)
    if api_base:
        config.api.api_base = api_base
    return config
