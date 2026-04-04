import os
import time
import requests
from dotenv import load_dotenv
from ratelimit import auth_limiter

load_dotenv()

_token_cache = {"access_token": None, "expires_at": 0}


def get_token():
    """Get a valid Bearer token, refreshing if expired (with 60s safety window)."""
    if _token_cache["access_token"] and time.time() < _token_cache["expires_at"] - 60:
        return _token_cache["access_token"]

    client_id = os.environ["EAGLEVIEW_CLIENT_ID"]
    client_secret = os.environ["EAGLEVIEW_CLIENT_SECRET"]
    token_url = os.environ["EAGLEVIEW_TOKEN_URL"]

    auth_limiter.wait()
    resp = requests.post(
        token_url,
        data={"grant_type": "client_credentials"},
        auth=(client_id, client_secret),
        headers={"Accept": "application/json"},
    )
    resp.raise_for_status()
    data = resp.json()

    _token_cache["access_token"] = data["access_token"]
    _token_cache["expires_at"] = time.time() + data.get("expires_in", 3600)

    print(f"[auth] Token acquired, expires in {data.get('expires_in', 3600)}s")
    return _token_cache["access_token"]


def auth_headers():
    """Return headers dict with Bearer token."""
    return {"Authorization": f"Bearer {get_token()}"}
