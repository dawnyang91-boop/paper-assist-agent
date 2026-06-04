from __future__ import annotations

from typing import Any, Optional

import httpx


class PaderAuthError(Exception):
    """Error raised when the upstream Pader auth service rejects a request."""

    def __init__(self, status_code: int, detail: Any) -> None:
        super().__init__(str(detail))
        self.status_code = status_code
        self.detail = detail


class PaderAuthClient:
    """HTTP client for the shared Pader account backend."""

    def __init__(self, base_url: str, timeout_seconds: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout_seconds,
            follow_redirects=True,
        )

    def register(
        self,
        name: str,
        email: str,
        phone: Optional[str],
        password: str,
        avatar_url: Optional[str] = None,
    ) -> dict[str, Any]:
        data = self._request(
            "POST",
            "/auth/register",
            json={
                "name": name,
                "email": email,
                "phone": phone,
                "password": password,
                "avatarUrl": avatar_url,
            },
        )
        return self._extract_user(data)

    def login(self, email: str, password: str) -> Optional[dict[str, Any]]:
        try:
            data = self._request(
                "POST",
                "/auth/login",
                json={"email": email, "password": password},
            )
        except PaderAuthError as exc:
            if exc.status_code in (400, 401, 403, 404):
                return None
            raise
        return self._extract_user(data)

    def get_user_profile(self, user_id: str) -> Optional[dict[str, Any]]:
        try:
            return self._request("GET", f"/users/{user_id}")
        except PaderAuthError as exc:
            if exc.status_code == 404:
                return None
            raise

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.RequestError as exc:
            raise PaderAuthError(
                502,
                {
                    "code": "pader_auth_unavailable",
                    "message": f"无法连接 Pader 账号服务：{exc}",
                },
            ) from exc

        if response.status_code >= 400:
            raise PaderAuthError(response.status_code, _response_detail(response))
        try:
            return response.json()
        except ValueError as exc:
            raise PaderAuthError(
                502,
                {
                    "code": "pader_auth_bad_response",
                    "message": "Pader 账号服务返回了无法解析的响应。",
                },
            ) from exc

    def _extract_user(self, data: Any) -> dict[str, Any]:
        if isinstance(data, dict) and isinstance(data.get("user"), dict):
            return data["user"]
        if isinstance(data, dict) and isinstance(data.get("id"), str):
            return data
        raise PaderAuthError(
            502,
            {
                "code": "pader_auth_bad_response",
                "message": "Pader 账号服务响应中缺少 user 对象。",
            },
        )


def _response_detail(response: httpx.Response) -> Any:
    try:
        data = response.json()
    except ValueError:
        return {
            "code": "pader_auth_http_error",
            "message": response.text or f"Pader auth request failed: {response.status_code}",
        }
    if isinstance(data, dict) and "detail" in data:
        return data["detail"]
    return data
