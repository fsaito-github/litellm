"""
OIDC (OpenID Connect) handler for LiteLLM Proxy SSO authentication.

Validates JWT tokens using JWKS keys fetched from an OIDC discovery endpoint,
and maps OIDC claims to LiteLLM user fields.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
import jwt as pyjwt
from jwt.api_jwk import PyJWK, PyJWKSet

from litellm._logging import verbose_proxy_logger

# Default OIDC → LiteLLM claim mapping
DEFAULT_CLAIM_MAPPING: Dict[str, str] = {
    "sub": "user_id",
    "email": "user_email",
    "groups": "teams",
    "roles": "user_role",
}

# JWKS cache TTL in seconds
_JWKS_CACHE_TTL: int = 3600


class OIDCHandler:
    """
    Handles OIDC-based SSO authentication for the LiteLLM proxy.

    Fetches OIDC discovery metadata, caches JWKS keys, validates JWT tokens,
    and maps OIDC claims to LiteLLM user fields.
    """

    def __init__(
        self,
        discovery_url: str,
        client_id: str,
        allowed_scopes: Optional[List[str]] = None,
        claim_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        self.discovery_url = discovery_url
        self.client_id = client_id
        self.allowed_scopes = allowed_scopes or ["openid", "profile", "email"]
        self.claim_mapping: Dict[str, str] = {
            **DEFAULT_CLAIM_MAPPING,
            **(claim_mapping or {}),
        }

        # Discovery metadata (populated by discover())
        self._issuer: Optional[str] = None
        self._jwks_uri: Optional[str] = None
        self._userinfo_endpoint: Optional[str] = None

        # JWKS cache
        self._jwks_keys: Optional[Dict[str, Any]] = None
        self._jwks_fetched_at: float = 0.0

    # ------------------------------------------------------------------
    # Discovery & JWKS
    # ------------------------------------------------------------------

    async def discover(self) -> Dict[str, Any]:
        """
        Fetch the OIDC discovery document and cache the JWKS keys.

        Returns:
            The parsed OIDC discovery document.

        Raises:
            Exception: If the discovery endpoint is unreachable or malformed.
        """
        verbose_proxy_logger.debug(
            "SSO OIDC: Fetching discovery document from %s", self.discovery_url
        )
        async with httpx.AsyncClient() as client:
            response = await client.get(self.discovery_url, timeout=10.0)

        if response.status_code != 200:
            raise Exception(
                f"SSO OIDC: Discovery endpoint {self.discovery_url} returned "
                f"status {response.status_code}: {response.text}"
            )

        discovery: Dict[str, Any] = response.json()

        self._issuer = discovery.get("issuer")
        self._jwks_uri = discovery.get("jwks_uri")
        self._userinfo_endpoint = discovery.get("userinfo_endpoint")

        if not self._jwks_uri:
            raise Exception(
                f"SSO OIDC: Discovery document at {self.discovery_url} "
                "does not contain a 'jwks_uri' field."
            )

        verbose_proxy_logger.info(
            "SSO OIDC: Discovered issuer=%s, jwks_uri=%s",
            self._issuer,
            self._jwks_uri,
        )

        await self._fetch_jwks()
        return discovery

    async def _fetch_jwks(self) -> None:
        """Fetch and cache JWKS keys from the jwks_uri."""
        if self._jwks_uri is None:
            raise Exception("SSO OIDC: jwks_uri not set. Call discover() first.")

        verbose_proxy_logger.debug(
            "SSO OIDC: Fetching JWKS from %s", self._jwks_uri
        )
        async with httpx.AsyncClient() as client:
            response = await client.get(self._jwks_uri, timeout=10.0)

        if response.status_code != 200:
            raise Exception(
                f"SSO OIDC: JWKS endpoint {self._jwks_uri} returned "
                f"status {response.status_code}: {response.text}"
            )

        self._jwks_keys = response.json()
        self._jwks_fetched_at = time.monotonic()
        verbose_proxy_logger.debug(
            "SSO OIDC: Cached %d JWKS keys",
            len(self._jwks_keys.get("keys", [])),
        )

    async def _get_jwks(self) -> Dict[str, Any]:
        """Return cached JWKS keys, refreshing if the TTL has expired."""
        now = time.monotonic()
        if (
            self._jwks_keys is None
            or (now - self._jwks_fetched_at) > _JWKS_CACHE_TTL
        ):
            if self._jwks_uri is None:
                await self.discover()
            else:
                await self._fetch_jwks()
        assert self._jwks_keys is not None
        return self._jwks_keys

    # ------------------------------------------------------------------
    # Token validation
    # ------------------------------------------------------------------

    def _resolve_signing_key(
        self, token: str, jwks_data: Dict[str, Any]
    ) -> Any:
        """Resolve the signing key for the given token from JWKS data."""
        header = pyjwt.get_unverified_header(token)
        kid = header.get("kid")

        jwk_set = PyJWKSet.from_dict(jwks_data)

        if kid:
            for key in jwk_set.keys:
                if key.key_id == kid:
                    return key.key
            raise Exception(
                f"SSO OIDC: No matching JWKS key found for kid={kid}"
            )

        # If no kid in header, use the first key
        if jwk_set.keys:
            return jwk_set.keys[0].key

        raise Exception("SSO OIDC: No keys available in JWKS endpoint")

    async def validate_token(self, token: str) -> Dict[str, Any]:
        """
        Validate a JWT token using cached JWKS keys and check standard claims.

        Args:
            token: The raw JWT token string.

        Returns:
            The decoded token payload as a dictionary.

        Raises:
            Exception: On expired, invalid, or malformed tokens.
        """
        jwks_data = await self._get_jwks()

        try:
            signing_key = self._resolve_signing_key(token, jwks_data)
        except Exception:
            # Key might have rotated — refresh once and retry
            verbose_proxy_logger.debug(
                "SSO OIDC: Key resolution failed, refreshing JWKS"
            )
            await self._fetch_jwks()
            jwks_data = await self._get_jwks()
            signing_key = self._resolve_signing_key(token, jwks_data)

        algorithms = [
            "RS256", "RS384", "RS512",
            "ES256", "ES384", "ES512",
            "PS256", "PS384", "PS512",
            "EdDSA",
        ]

        decode_options: Dict[str, Any] = {}
        audience: Optional[str] = self.client_id

        try:
            payload: Dict[str, Any] = pyjwt.decode(
                token,
                signing_key,
                algorithms=algorithms,
                audience=audience,
                options=decode_options,
            )
        except pyjwt.ExpiredSignatureError:
            raise Exception("SSO OIDC: Token has expired")
        except pyjwt.InvalidAudienceError:
            raise Exception(
                f"SSO OIDC: Token audience does not match client_id={self.client_id}"
            )
        except pyjwt.InvalidIssuerError:
            raise Exception("SSO OIDC: Token issuer is invalid")
        except pyjwt.DecodeError as exc:
            raise Exception(f"SSO OIDC: Failed to decode token — {exc}")
        except Exception as exc:
            raise Exception(f"SSO OIDC: Token validation failed — {exc}")

        # Verify issuer matches discovery if available
        if self._issuer and payload.get("iss") != self._issuer:
            raise Exception(
                f"SSO OIDC: Token issuer '{payload.get('iss')}' does not match "
                f"discovered issuer '{self._issuer}'"
            )

        verbose_proxy_logger.debug(
            "SSO OIDC: Token validated for sub=%s", payload.get("sub")
        )
        return payload

    # ------------------------------------------------------------------
    # Claim mapping
    # ------------------------------------------------------------------

    def map_claims_to_user(self, claims: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map OIDC token claims to LiteLLM user fields using the configured
        claim mapping.

        Default mapping:
            sub          → user_id
            email        → user_email
            groups       → teams  (list of team names)
            roles        → user_role

        Args:
            claims: Decoded JWT payload.

        Returns:
            A dict with LiteLLM user field names as keys.
        """
        user_fields: Dict[str, Any] = {}
        for oidc_claim, litellm_field in self.claim_mapping.items():
            value = claims.get(oidc_claim)
            if value is not None:
                user_fields[litellm_field] = value

        verbose_proxy_logger.debug(
            "SSO OIDC: Mapped claims to user fields: %s",
            list(user_fields.keys()),
        )
        return user_fields

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """No persistent resources to clean up."""
        pass
