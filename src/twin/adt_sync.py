"""Azure Digital Twins mirror for the in-process physics twin.

This module is the *only* place that talks to ADT. The rest of the codebase
keeps working when ADT is not configured — :func:`adt_mirror` returns a
``NullMirror`` that no-ops with a clear log message. Authentication uses
``DefaultAzureCredential`` so the same code path works locally (developer
``az login``) and in Container Apps (system-assigned MSI).

Configuration env vars:

- ``ADT_ENDPOINT``  – full hostname, e.g. ``adt-tra.api.weu.digitaltwins.azure.net``
- ``ADT_MODEL_ID``  – DTDL model id (default ``dtmi:tra:Transformer;1``)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "dtmi:tra:Transformer;1"


class TwinMirror(Protocol):
    enabled: bool
    endpoint: str

    def upsert_twin(self, asset_id: str, state: dict[str, Any], telemetry: dict[str, Any] | None = None) -> dict[str, Any]: ...

    def get_twin(self, asset_id: str) -> dict[str, Any] | None: ...

    def query(self, adt_query: str) -> list[dict[str, Any]]: ...

    def ensure_model(self, dtdl_json: dict[str, Any]) -> None: ...


@dataclass
class NullMirror:
    """No-op mirror used when ADT is not configured."""

    endpoint: str = ""
    enabled: bool = False
    reason: str = "ADT_ENDPOINT not set"

    def upsert_twin(self, asset_id: str, state: dict[str, Any], telemetry: dict[str, Any] | None = None) -> dict[str, Any]:
        logger.info("NullMirror.upsert_twin(%s) skipped — %s", asset_id, self.reason)
        return {"skipped": True, "reason": self.reason}

    def get_twin(self, asset_id: str) -> dict[str, Any] | None:
        return None

    def query(self, adt_query: str) -> list[dict[str, Any]]:
        return []

    def ensure_model(self, dtdl_json: dict[str, Any]) -> None:
        return None


class AzureDigitalTwinsMirror:
    """Real Azure Digital Twins mirror backed by ``azure-digitaltwins-core``."""

    enabled = True

    def __init__(self, endpoint: str, model_id: str = DEFAULT_MODEL_ID) -> None:
        try:
            from azure.digitaltwins.core import DigitalTwinsClient
            from azure.identity import DefaultAzureCredential
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "azure-digitaltwins-core is not installed. Add it to requirements.txt."
            ) from exc
        self.endpoint = endpoint if endpoint.startswith("https://") else f"https://{endpoint}"
        self.model_id = model_id
        self._client = DigitalTwinsClient(self.endpoint, DefaultAzureCredential())

    def ensure_model(self, dtdl_json: dict[str, Any]) -> None:
        from azure.core.exceptions import HttpResponseError, ResourceNotFoundError

        try:
            self._client.get_model(self.model_id)
            return
        except ResourceNotFoundError:
            pass
        except HttpResponseError as exc:  # pragma: no cover
            logger.warning("ADT get_model failed: %s", exc)
        try:
            self._client.create_models([dtdl_json])
            logger.info("Uploaded DTDL model %s", self.model_id)
        except HttpResponseError as exc:  # pragma: no cover
            logger.warning("ADT create_models failed: %s", exc)

    def upsert_twin(self, asset_id: str, state: dict[str, Any], telemetry: dict[str, Any] | None = None) -> dict[str, Any]:
        from azure.core.exceptions import HttpResponseError

        payload = {"$metadata": {"$model": self.model_id}}
        for key, value in state.items():
            if value is None:
                continue
            payload[key] = value
        try:
            self._client.upsert_digital_twin(asset_id, payload)
        except HttpResponseError as exc:  # pragma: no cover
            logger.warning("ADT upsert failed: %s", exc)
            return {"error": str(exc)}
        if telemetry:
            try:
                self._client.publish_telemetry(asset_id, telemetry)
            except HttpResponseError as exc:  # pragma: no cover
                logger.warning("ADT publish_telemetry failed: %s", exc)
        return {"ok": True, "twin_id": asset_id}

    def get_twin(self, asset_id: str) -> dict[str, Any] | None:
        from azure.core.exceptions import ResourceNotFoundError

        try:
            return dict(self._client.get_digital_twin(asset_id))
        except ResourceNotFoundError:
            return None
        except Exception as exc:  # pragma: no cover
            logger.warning("ADT get_digital_twin failed: %s", exc)
            return None

    def query(self, adt_query: str) -> list[dict[str, Any]]:
        try:
            results = self._client.query_twins(adt_query)
            return [dict(item) for item in results]
        except Exception as exc:  # pragma: no cover
            logger.warning("ADT query failed: %s", exc)
            return []


_singleton: TwinMirror | None = None


def adt_mirror() -> TwinMirror:
    """Return a process-wide mirror (real if ``ADT_ENDPOINT`` is set, else null)."""
    global _singleton
    if _singleton is not None:
        return _singleton
    endpoint = os.environ.get("ADT_ENDPOINT", "").strip()
    model_id = os.environ.get("ADT_MODEL_ID", DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID
    if not endpoint:
        _singleton = NullMirror()
        return _singleton
    try:
        _singleton = AzureDigitalTwinsMirror(endpoint=endpoint, model_id=model_id)
    except Exception as exc:  # pragma: no cover
        logger.warning("ADT mirror unavailable, falling back to NullMirror: %s", exc)
        _singleton = NullMirror(reason=f"init failed: {exc}")
    return _singleton


def reset_mirror_for_tests() -> None:
    global _singleton
    _singleton = None
