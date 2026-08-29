from fastapi.testclient import TestClient


def _create_session(client: TestClient, name: str | None = None) -> dict:
    payload = {"name": name} if name is not None else {}
    response = client.post("/sessions", json=payload)
    assert response.status_code == 201
    return response.json()


def test_create_session_defaults_name(client: TestClient) -> None:
    session = _create_session(client)
    assert session["name"] == "New session"
    assert "id" in session and "created_at" in session and "updated_at" in session


def test_create_session_with_custom_name(client: TestClient) -> None:
    session = _create_session(client, name="Login feature")
    assert session["name"] == "Login feature"


def test_list_sessions_orders_most_recently_updated_first(client: TestClient) -> None:
    first = _create_session(client, name="First")
    second = _create_session(client, name="Second")

    response = client.get("/sessions")
    assert response.status_code == 200
    ids = [s["id"] for s in response.json()]
    assert ids.index(second["id"]) < ids.index(first["id"])


def test_get_session_returns_empty_history_right_after_creation(client: TestClient) -> None:
    session = _create_session(client)
    response = client.get(f"/sessions/{session['id']}")
    assert response.status_code == 200
    assert response.json()["messages"] == []


def test_get_session_404_for_missing_session(client: TestClient) -> None:
    response = client.get("/sessions/9999")
    assert response.status_code == 404


def test_rename_session(client: TestClient) -> None:
    session = _create_session(client, name="Old name")
    response = client.patch(f"/sessions/{session['id']}", json={"name": "New name"})
    assert response.status_code == 200
    assert response.json()["name"] == "New name"


def test_rename_session_404_for_missing_session(client: TestClient) -> None:
    response = client.patch("/sessions/9999", json={"name": "New name"})
    assert response.status_code == 404


def test_delete_session(client: TestClient) -> None:
    session = _create_session(client)
    response = client.delete(f"/sessions/{session['id']}")
    assert response.status_code == 204
    assert client.get(f"/sessions/{session['id']}").status_code == 404


def test_delete_session_404_for_missing_session(client: TestClient) -> None:
    response = client.delete("/sessions/9999")
    assert response.status_code == 404


def test_post_message_returns_canned_assistant_reply(client: TestClient) -> None:
    session = _create_session(client)
    response = client.post(
        f"/sessions/{session['id']}/messages", json={"text": "Hello", "attachments": []}
    )
    assert response.status_code == 201
    body = response.json()
    assert body["role"] == "assistant"
    assert isinstance(body["text"], str) and body["text"]


def test_post_message_404_for_missing_session(client: TestClient) -> None:
    response = client.post("/sessions/9999/messages", json={"text": "Hello", "attachments": []})
    assert response.status_code == 404


def test_message_history_persists_user_and_assistant_messages_in_order(
    client: TestClient,
) -> None:
    session = _create_session(client)
    client.post(f"/sessions/{session['id']}/messages", json={"text": "Hello", "attachments": []})

    detail = client.get(f"/sessions/{session['id']}").json()
    roles = [m["role"] for m in detail["messages"]]
    assert roles == ["user", "assistant"]
    assert detail["messages"][0]["text"] == "Hello"


def test_generate_returns_example_acceptance_criteria(client: TestClient) -> None:
    session = _create_session(client)
    client.post(
        f"/sessions/{session['id']}/messages",
        json={
            "text": "Attached are the user stories",
            "attachments": [{"filename": "UserStorySet1.txt", "content": "As a user..."}],
        },
    )

    response = client.post(f"/sessions/{session['id']}/generate")
    assert response.status_code == 200
    body = response.json()
    assert len(body["acceptance_criteria"]) >= 1
    first = body["acceptance_criteria"][0]
    assert {
        "id",
        "title",
        "given",
        "when",
        "then",
        "scores",
        "overall_score",
        "status",
    } <= first.keys()
    assert {"relevance", "correctness", "understandability", "coverage"} <= first[
        "scores"
    ].keys()
    assert first["status"] == "pending"


def _without_ids(acceptance_criteria: list[dict]) -> list[dict]:
    return [{k: v for k, v in item.items() if k != "id"} for item in acceptance_criteria]


def test_generate_ignores_conversation_content(client: TestClient) -> None:
    empty_session = _create_session(client)
    populated_session = _create_session(client)
    client.post(
        f"/sessions/{populated_session['id']}/messages",
        json={"text": "completely different content", "attachments": []},
    )

    response_a = client.post(f"/sessions/{empty_session['id']}/generate")
    response_b = client.post(f"/sessions/{populated_session['id']}/generate")
    # Content is identical regardless of input, but ids are real per-session
    # auto-increment values now, so they legitimately differ between sessions.
    assert _without_ids(response_a.json()["acceptance_criteria"]) == _without_ids(
        response_b.json()["acceptance_criteria"]
    )


def test_generate_404_for_missing_session(client: TestClient) -> None:
    response = client.post("/sessions/9999/generate")
    assert response.status_code == 404
