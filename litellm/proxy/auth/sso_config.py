"""
SSO Configuration models and factory for LiteLLM Proxy.

Supports OIDC and Entra ID (Azure AD) authentication providers.
"""

from typing import TYPE_CHECKING, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from litellm._logging import verbose_proxy_logger

if TYPE_CHECKING:
    from litellm.proxy.auth.entra_id_handler import EntraIDHandler
    from litellm.proxy.auth.oidc_handler import OIDCHandler


class SSOConfig(BaseModel):
    """Configuration model for SSO authentication."""

    provider: Literal["oidc", "entra_id"] = "oidc"
    discovery_url: Optional[str] = None
    client_id: str
    client_secret: Optional[str] = None
    tenant_id: Optional[str] = Field(
        default=None, description="Azure AD tenant ID. Required for Entra ID provider."
    )
    allowed_scopes: List[str] = Field(
        default=["openid", "profile", "email"],
        description="OAuth2 scopes to accept.",
    )
    claim_mapping: Optional[Dict[str, str]] = Field(
        default=None,
        description="Custom mapping from OIDC claims to LiteLLM user fields.",
    )
    mfa_required_for: List[str] = Field(
        default=[],
        description="Roles or actions that require MFA verification.",
    )
    session_timeout_minutes: int = Field(
        default=480, description="Session timeout in minutes."
    )
    max_concurrent_sessions: int = Field(
        default=5, description="Maximum concurrent sessions per user."
    )

    @model_validator(mode="after")
    def _validate_provider_fields(self) -> "SSOConfig":
        if self.provider == "entra_id" and not self.tenant_id:
            raise ValueError("tenant_id is required when provider is 'entra_id'")
        if self.provider == "oidc" and not self.discovery_url:
            raise ValueError("discovery_url is required when provider is 'oidc'")
        return self


def create_sso_handler(
    config: SSOConfig,
) -> "OIDCHandler | EntraIDHandler":
    """
    Factory function that returns the appropriate SSO handler based on config.

    Args:
        config: The SSO configuration.

    Returns:
        An OIDCHandler or EntraIDHandler instance.
    """
    verbose_proxy_logger.info(
        "SSO: Creating handler for provider=%s", config.provider
    )

    if config.provider == "entra_id":
        from litellm.proxy.auth.entra_id_handler import EntraIDHandler

        return EntraIDHandler(
            tenant_id=config.tenant_id,  # type: ignore[arg-type]
            client_id=config.client_id,
            client_secret=config.client_secret,
        )

    from litellm.proxy.auth.oidc_handler import OIDCHandler

    return OIDCHandler(
        discovery_url=config.discovery_url,  # type: ignore[arg-type]
        client_id=config.client_id,
        allowed_scopes=config.allowed_scopes,
        claim_mapping=config.claim_mapping,
    )
