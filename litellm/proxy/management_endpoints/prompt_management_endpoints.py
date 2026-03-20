"""
PROMPT MANAGEMENT

All /prompts management endpoints

POST   /prompts                              - Create a new prompt
GET    /prompts                              - List all prompts
GET    /prompts/{prompt_name}                - Get prompt by name
PUT    /prompts/{prompt_name}                - Update prompt (creates new version)
GET    /prompts/{prompt_name}/versions       - List all versions
POST   /prompts/{prompt_name}/render         - Render prompt with variables
POST   /prompts/{prompt_name}/rollback/{version} - Rollback to specific version
"""

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()

# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class CreatePromptRequest(BaseModel):
    prompt_name: str
    template: str
    description: Optional[str] = None
    variables: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class UpdatePromptRequest(BaseModel):
    template: str
    description: Optional[str] = None
    variables: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    change_note: Optional[str] = None


class RenderPromptRequest(BaseModel):
    variables: Dict[str, str]


class PromptResponse(BaseModel):
    prompt_id: str
    prompt_name: str
    description: Optional[str] = None
    template: str
    variables: List[str] = Field(default_factory=list)
    version: int
    is_active: bool
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str
    created_by: str
    updated_at: str
    updated_by: str


class PromptVersionResponse(BaseModel):
    id: str
    prompt_id: str
    version: int
    template: str
    variables: List[str] = Field(default_factory=list)
    change_note: Optional[str] = None
    created_at: str
    created_by: str


class PromptListResponse(BaseModel):
    prompts: List[PromptResponse]
    total: int
    page: int
    page_size: int


class RenderPromptResponse(BaseModel):
    rendered: str
    prompt_name: str
    version: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VARIABLE_PATTERN = re.compile(r"\{\{(\w+)\}\}")


def _extract_variables(template: str) -> List[str]:
    """Return unique variable names found in a ``{{var}}`` template."""
    return list(dict.fromkeys(_VARIABLE_PATTERN.findall(template)))


def _prompt_record_to_response(record: Any) -> PromptResponse:
    variables_raw = record.variables
    if isinstance(variables_raw, str):
        variables_raw = json.loads(variables_raw)

    metadata_raw = record.metadata
    if isinstance(metadata_raw, str):
        metadata_raw = json.loads(metadata_raw)

    return PromptResponse(
        prompt_id=record.prompt_id,
        prompt_name=record.prompt_name,
        description=record.description,
        template=record.template,
        variables=variables_raw or [],
        version=record.version,
        is_active=record.is_active,
        metadata=metadata_raw or {},
        created_at=record.created_at.isoformat(),
        created_by=record.created_by,
        updated_at=record.updated_at.isoformat(),
        updated_by=record.updated_by,
    )


def _version_record_to_response(record: Any) -> PromptVersionResponse:
    variables_raw = record.variables
    if isinstance(variables_raw, str):
        variables_raw = json.loads(variables_raw)

    return PromptVersionResponse(
        id=record.id,
        prompt_id=record.prompt_id,
        version=record.version,
        template=record.template,
        variables=variables_raw or [],
        change_note=record.change_note,
        created_at=record.created_at.isoformat(),
        created_by=record.created_by,
    )


def _get_user_id(user_api_key_dict: UserAPIKeyAuth) -> str:
    return user_api_key_dict.user_id or "unknown"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/prompts",
    tags=["prompt management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PromptResponse,
)
async def create_prompt(
    data: CreatePromptRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Create a new prompt template.

    Auto-detects ``{{variable}}`` placeholders from the template.
    Also creates version 1 in the version history table.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail=CommonProxyErrors.db_not_connected_error.value,
        )

    try:
        # Check for duplicate name
        existing = await prisma_client.db.litellm_prompttable.find_unique(
            where={"prompt_name": data.prompt_name}
        )
        if existing is not None:
            raise HTTPException(
                status_code=400,
                detail=f"Prompt with name '{data.prompt_name}' already exists",
            )

        # Auto-detect variables from template
        detected_vars = _extract_variables(data.template)
        variables = data.variables if data.variables is not None else detected_vars

        user_id = _get_user_id(user_api_key_dict)

        # Create the prompt record
        new_prompt = await prisma_client.db.litellm_prompttable.create(
            data={
                "prompt_name": data.prompt_name,
                "template": data.template,
                "description": data.description or "",
                "variables": json.dumps(variables),
                "version": 1,
                "is_active": True,
                "metadata": json.dumps(data.metadata or {}),
                "created_by": user_id,
                "updated_by": user_id,
            }
        )

        # Create version 1 entry
        await prisma_client.db.litellm_promptversiontable.create(
            data={
                "prompt_id": new_prompt.prompt_id,
                "version": 1,
                "template": data.template,
                "variables": json.dumps(variables),
                "change_note": "Initial version",
                "created_by": user_id,
            }
        )

        return _prompt_record_to_response(new_prompt)

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error creating prompt: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/prompts",
    tags=["prompt management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PromptListResponse,
)
async def list_prompts(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=25, ge=1, le=100, description="Items per page"),
    is_active: Optional[bool] = Query(default=None, description="Filter by active status"),
):
    """List all prompts with pagination and optional active-status filter."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail=CommonProxyErrors.db_not_connected_error.value,
        )

    try:
        where: Dict[str, Any] = {}
        if is_active is not None:
            where["is_active"] = is_active

        total = await prisma_client.db.litellm_prompttable.count(where=where)

        records = await prisma_client.db.litellm_prompttable.find_many(
            where=where,
            skip=(page - 1) * page_size,
            take=page_size,
            order={"created_at": "desc"},
        )

        return PromptListResponse(
            prompts=[_prompt_record_to_response(r) for r in records],
            total=total,
            page=page,
            page_size=page_size,
        )

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error listing prompts: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/prompts/{prompt_name}",
    tags=["prompt management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PromptResponse,
)
async def get_prompt(
    prompt_name: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Get the latest active version of a prompt by name."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail=CommonProxyErrors.db_not_connected_error.value,
        )

    try:
        record = await prisma_client.db.litellm_prompttable.find_unique(
            where={"prompt_name": prompt_name}
        )
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=f"Prompt '{prompt_name}' not found",
            )

        return _prompt_record_to_response(record)

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error getting prompt: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.put(
    "/prompts/{prompt_name}",
    tags=["prompt management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PromptResponse,
)
async def update_prompt(
    prompt_name: str,
    data: UpdatePromptRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Update a prompt template.

    Creates a new version automatically and stores the previous
    version in the version history table.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail=CommonProxyErrors.db_not_connected_error.value,
        )

    try:
        existing = await prisma_client.db.litellm_prompttable.find_unique(
            where={"prompt_name": prompt_name}
        )
        if existing is None:
            raise HTTPException(
                status_code=404,
                detail=f"Prompt '{prompt_name}' not found",
            )

        new_version = existing.version + 1

        detected_vars = _extract_variables(data.template)
        variables = data.variables if data.variables is not None else detected_vars

        user_id = _get_user_id(user_api_key_dict)

        # Build update payload
        update_data: Dict[str, Any] = {
            "template": data.template,
            "variables": json.dumps(variables),
            "version": new_version,
            "updated_by": user_id,
        }
        if data.description is not None:
            update_data["description"] = data.description
        if data.metadata is not None:
            update_data["metadata"] = json.dumps(data.metadata)

        updated = await prisma_client.db.litellm_prompttable.update(
            where={"prompt_name": prompt_name},
            data=update_data,
        )

        # Store the new version in the version history
        await prisma_client.db.litellm_promptversiontable.create(
            data={
                "prompt_id": existing.prompt_id,
                "version": new_version,
                "template": data.template,
                "variables": json.dumps(variables),
                "change_note": data.change_note,
                "created_by": user_id,
            }
        )

        return _prompt_record_to_response(updated)

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error updating prompt: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/prompts/{prompt_name}/versions",
    tags=["prompt management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=List[PromptVersionResponse],
)
async def list_prompt_versions(
    prompt_name: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """List all versions of a prompt."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail=CommonProxyErrors.db_not_connected_error.value,
        )

    try:
        prompt = await prisma_client.db.litellm_prompttable.find_unique(
            where={"prompt_name": prompt_name}
        )
        if prompt is None:
            raise HTTPException(
                status_code=404,
                detail=f"Prompt '{prompt_name}' not found",
            )

        versions = await prisma_client.db.litellm_promptversiontable.find_many(
            where={"prompt_id": prompt.prompt_id},
            order={"version": "desc"},
        )

        return [_version_record_to_response(v) for v in versions]

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error listing prompt versions: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/prompts/{prompt_name}/render",
    tags=["prompt management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=RenderPromptResponse,
)
async def render_prompt(
    prompt_name: str,
    data: RenderPromptRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Render a prompt by substituting ``{{variable}}`` placeholders."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail=CommonProxyErrors.db_not_connected_error.value,
        )

    try:
        prompt = await prisma_client.db.litellm_prompttable.find_unique(
            where={"prompt_name": prompt_name}
        )
        if prompt is None:
            raise HTTPException(
                status_code=404,
                detail=f"Prompt '{prompt_name}' not found",
            )

        rendered = prompt.template
        for key, value in data.variables.items():
            rendered = rendered.replace("{{" + key + "}}", value)

        # Warn about unresolved placeholders
        unresolved = _VARIABLE_PATTERN.findall(rendered)
        if unresolved:
            verbose_proxy_logger.warning(
                "Unresolved variables in prompt '%s': %s",
                prompt_name,
                unresolved,
            )

        return RenderPromptResponse(
            rendered=rendered,
            prompt_name=prompt_name,
            version=prompt.version,
        )

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error rendering prompt: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/prompts/{prompt_name}/rollback/{version}",
    tags=["prompt management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PromptResponse,
)
async def rollback_prompt(
    prompt_name: str,
    version: int,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Rollback a prompt to a specific historical version.

    This creates a *new* version whose content matches the target
    historical version, preserving full audit history.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail=CommonProxyErrors.db_not_connected_error.value,
        )

    try:
        prompt = await prisma_client.db.litellm_prompttable.find_unique(
            where={"prompt_name": prompt_name}
        )
        if prompt is None:
            raise HTTPException(
                status_code=404,
                detail=f"Prompt '{prompt_name}' not found",
            )

        # Find the requested version
        target_version = await prisma_client.db.litellm_promptversiontable.find_first(
            where={
                "prompt_id": prompt.prompt_id,
                "version": version,
            }
        )
        if target_version is None:
            raise HTTPException(
                status_code=404,
                detail=f"Version {version} not found for prompt '{prompt_name}'",
            )

        new_version = prompt.version + 1
        user_id = _get_user_id(user_api_key_dict)

        variables_raw = target_version.variables
        if isinstance(variables_raw, str):
            variables_raw = json.loads(variables_raw)

        # Update the main prompt record to the rolled-back content
        updated = await prisma_client.db.litellm_prompttable.update(
            where={"prompt_name": prompt_name},
            data={
                "template": target_version.template,
                "variables": json.dumps(variables_raw or []),
                "version": new_version,
                "updated_by": user_id,
            },
        )

        # Record the rollback as a new version entry
        await prisma_client.db.litellm_promptversiontable.create(
            data={
                "prompt_id": prompt.prompt_id,
                "version": new_version,
                "template": target_version.template,
                "variables": json.dumps(variables_raw or []),
                "change_note": f"Rollback to version {version}",
                "created_by": user_id,
            }
        )

        return _prompt_record_to_response(updated)

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error rolling back prompt: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
