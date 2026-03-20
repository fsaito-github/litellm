"""Unit tests for litellm.proxy.hooks.compliance_latam module."""

import pytest

from litellm.proxy.hooks.compliance_latam import (
    ComplianceConfig,
    DataResidencyHook,
    PIIMaskingHook,
)


# -------------------------------------------------------------------
# PIIMaskingHook — mask_pii
# -------------------------------------------------------------------


class TestPIIMasking:
    @pytest.fixture
    def hook(self):
        return PIIMaskingHook()

    def test_cpf_formatted_masking(self, hook):
        text = "My CPF is 123.456.789-01"
        masked, detections = hook.mask_pii(text)
        assert "[CPF:***.***.***-**]" in masked
        assert "123.456.789-01" not in masked
        assert len(detections) >= 1
        assert any(d["type"] == "cpf" for d in detections)

    def test_cnpj_masking(self, hook):
        text = "CNPJ: 12.345.678/0001-90"
        masked, detections = hook.mask_pii(text)
        assert "[CNPJ:MASKED]" in masked
        assert "12.345.678/0001-90" not in masked

    def test_phone_masking(self, hook):
        text = "Call me at +55 (11) 98765-4321"
        masked, detections = hook.mask_pii(text)
        assert "[PHONE:MASKED]" in masked
        assert any(d["type"] == "phone" for d in detections)

    def test_credit_card_masking(self, hook):
        text = "Card: 4111 1111 1111 1111"
        masked, detections = hook.mask_pii(text)
        assert "[CREDIT_CARD:****-****-****-****]" in masked

    def test_email_masking(self, hook):
        text = "Email: user@example.com"
        masked, detections = hook.mask_pii(text)
        assert "[EMAIL:MASKED]" in masked
        assert "user@example.com" not in masked

    def test_cpf_without_dots(self, hook):
        text = "CPF raw: 12345678901"
        masked, detections = hook.mask_pii(text)
        assert "[CPF:***.***.***-**]" in masked
        assert any(d["type"] == "cpf" for d in detections)

    def test_multiple_pii_in_same_text(self, hook):
        text = "CPF: 123.456.789-01, email: user@test.com"
        masked, detections = hook.mask_pii(text)
        assert "[CPF:***.***.***-**]" in masked
        assert "[EMAIL:MASKED]" in masked
        types = {d["type"] for d in detections}
        assert "cpf" in types
        assert "email" in types

    def test_clean_text_returns_unchanged(self, hook):
        text = "Hello, this is a normal sentence with no PII."
        masked, detections = hook.mask_pii(text)
        assert masked == text
        assert len(detections) == 0


# -------------------------------------------------------------------
# DataResidencyHook — check_model_region
# -------------------------------------------------------------------


class TestDataResidencyHook:
    @pytest.fixture
    def hook(self):
        return DataResidencyHook(
            allowed_regions=["brazilsouth", "brazilsoutheast"]
        )

    def test_allowed_region_passes(self, hook):
        result = hook.check_model_region(
            "gpt-4",
            {
                "api_base": (
                    "https://myresource.brazilsouth.api.cognitive.microsoft.com/"
                )
            },
        )
        assert result is True

    def test_blocked_region(self, hook):
        result = hook.check_model_region(
            "gpt-4",
            {
                "api_base": (
                    "https://myresource.eastus.api.cognitive.microsoft.com/"
                )
            },
        )
        assert result is False

    def test_no_api_base_non_compliant(self, hook):
        """When no api_base is provided the region cannot be determined;
        the hook treats this as non-compliant."""
        result = hook.check_model_region("gpt-4", {})
        assert result is False

    def test_explicit_region_param_allowed(self, hook):
        result = hook.check_model_region(
            "gpt-4", {"region": "brazilsouth"}
        )
        assert result is True

    def test_explicit_region_param_blocked(self, hook):
        result = hook.check_model_region(
            "gpt-4", {"region": "eastus"}
        )
        assert result is False


# -------------------------------------------------------------------
# ComplianceConfig
# -------------------------------------------------------------------


class TestComplianceConfig:
    def test_default_values(self):
        config = ComplianceConfig()
        assert config.enabled is False
        assert config.pii_masking is True
        assert config.data_residency is False
        assert "brazilsouth" in config.allowed_regions
        assert "brazilsoutheast" in config.allowed_regions
        assert config.audit_retention_days == 1825
        assert config.mask_direction == "both"

    def test_custom_values(self):
        config = ComplianceConfig(
            enabled=True,
            pii_masking=True,
            data_residency=True,
            allowed_regions=["brazilsouth"],
        )
        assert config.enabled is True
        assert config.data_residency is True
        assert config.allowed_regions == ["brazilsouth"]
