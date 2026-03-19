"""
Entra ID (Azure AD) handler for LiteLLM Proxy SSO authentication.

Extends the generic OIDCHandler with Azure-specific claim mappings,
MFA detection, and tenant validation.
"""

from typing import Any, Dict, List, Optional

from litellm._logging import verbose_proxy_logger
from litellm.proxy.auth.oidc_handler import OIDCHandler

# Entra ID specific claim mapping
_ENTRA_CLAIM_MAPPING: Dict[str, str] = {
    "preferred_username": "user_email",
    "oid": "user_id",
    "sub": "user_id",  # fallback
    "groups": "teams",
    "wids": "roles",
}


class EntraIDHandler(OIDCHandler):
    """
    Azure Entra ID (Azure AD) SSO handler.

    Auto-constructs the OIDC discovery URL from the tenant ID and applies
    Entra ID–specific claim mappings.
    """

    def __init__(
        self,
        tenant_id: str,
        client_id: str,
        client_secret: Optional[str] = None,
    ) -> None:
        self.tenant_id = tenant_id
        self._client_secret = client_secret

        discovery_url = (
            f"https://login.microsoftonline.com/{tenant_id}"
            "/v2.0/.well-known/openid-configuration"
        )

        super().__init__(
            discovery_url=discovery_url,
            client_id=client_id,
            allowed_scopes=["openid", "profile", "email"],
            claim_mapping=_ENTRA_CLAIM_MAPPING,
        )

        verbose_proxy_logger.info(
            "SSO Entra ID: Initialised for tenant=%s, client_id=%s",
            tenant_id,
            client_id,
        )

    # ------------------------------------------------------------------
    # Entra ID–specific claim helpers
    # ------------------------------------------------------------------

    def map_claims_to_user(self, claims: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map Entra ID claims to LiteLLM user fields.

        Entra ID uses ``oid`` as the stable user identifier (falling back to
        ``sub``), ``preferred_username`` for email, ``groups`` for Azure AD
        group IDs, and ``wids`` for directory role IDs.
        """
        user_fields: Dict[str, Any] = {}

        # user_id: prefer oid, fall back to sub
        user_fields["user_id"] = claims.get("oid") or claims.get("sub")

        # user_email: prefer preferred_username, fall back to email
        email = claims.get("preferred_username") or claims.get("email")
        if email:
            user_fields["user_email"] = email

        # teams: Azure AD group IDs
        groups = claims.get("groups")
        if groups:
            user_fields["teams"] = groups if isinstance(groups, list) else [groups]

        # roles: directory role template IDs
        wids = claims.get("wids")
        if wids:
            user_fields["roles"] = wids if isinstance(wids, list) else [wids]

        # Also check standard 'roles' claim (app roles assigned in Azure)
        app_roles = claims.get("roles")
        if app_roles:
            existing = user_fields.get("roles", [])
            if isinstance(app_roles, list):
                user_fields["roles"] = existing + app_roles
            else:
                user_fields["roles"] = existing + [app_roles]

        verbose_proxy_logger.debug(
            "SSO Entra ID: Mapped claims to user fields: %s",
            list(user_fields.keys()),
        )
        return user_fields

    # ------------------------------------------------------------------
    # MFA & tenant helpers
    # ------------------------------------------------------------------

    @staticmethod
    def has_mfa(claims: Dict[str, Any]) -> bool:
        """
        Check if the token indicates multi-factor authentication was used.

        Azure Entra ID includes an ``amr`` (Authentication Methods References)
        claim. If ``'mfa'`` is present in that list, the user completed MFA.

        Args:
            claims: Decoded JWT payload.

        Returns:
            True if MFA was used, False otherwise.
        """
        amr = claims.get("amr", [])
        return "mfa" in amr

    @staticmethod
    def get_tenant_id(claims: Dict[str, Any]) -> Optional[str]:
        """
        Extract the Azure AD tenant ID from the ``tid`` claim.

        Args:
            claims: Decoded JWT payload.

        Returns:
            The tenant ID string, or None if the claim is absent.
        """
        return claims.get("tid")
