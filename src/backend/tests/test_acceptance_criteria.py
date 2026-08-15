from fastapi.testclient import TestClient


def _create_session(client: TestClient) -> dict:
    response = client.post("/sessions", json={})
    assert response.status_code == 201
    return response.json()


def _generate(client: TestClient, session_id: int) -> list[dict]:
    response = client.post(f"/sessions/{session_id}/generate")
    assert response.status_code == 200
    return response.json()["acceptance_criteria"]


def test_generate_persists_items_with_real_ids_and_pending_status(client: TestClient) -> None:
    session = _create_session(client)
    items = _generate(client, session["id"])
    assert len(items) >= 1
    assert all(item["status"] == "pending" for item in items)
    assert all(isinstance(item["id"], int) for item in items)

    listed = client.get(f"/sessions/{session['id']}/acceptance-criteria")
    assert listed.status_code == 200
    assert listed.json()["acceptance_criteria"] == items


def test_generate_replaces_existing_batch(client: TestClient) -> None:
    session = _create_session(client)
    _generate(client, session["id"])
    second = _generate(client, session["id"])

    # The second generate call should replace the first batch, not append to
    # it — confirm the persisted list matches the second call exactly rather
    # than containing both batches' worth of items.
    listed = client.get(f"/sessions/{session['id']}/acceptance-criteria").json()
    assert listed["acceptance_criteria"] == second


def test_list_acceptance_criteria_404_for_missing_session(client: TestClient) -> None:
    response = client.get("/sessions/9999/acceptance-criteria")
    assert response.status_code == 404


def test_update_text_saves_edit_and_leaves_other_fields_untouched(client: TestClient) -> None:
    session = _create_session(client)
    items = _generate(client, session["id"])
    target = items[0]

    response = client.patch(
        f"/sessions/{session['id']}/acceptance-criteria/{target['id']}",
        json={
            "title": "Edited title",
            "given": "edited given",
            "when": "edited when",
            "then": "edited then",
        },
    )
    assert response.status_code == 200
    updated = response.json()
    assert updated["title"] == "Edited title"
    assert updated["given"] == "edited given"
    assert updated["scores"] == target["scores"]
    assert updated["overall_score"] == target["overall_score"]
    assert updated["status"] == target["status"]


def test_update_text_404_for_missing_ac_or_session(client: TestClient) -> None:
    session = _create_session(client)
    _generate(client, session["id"])
    body = {"title": "t", "given": "g", "when": "w", "then": "t"}

    assert (
        client.patch(f"/sessions/{session['id']}/acceptance-criteria/9999", json=body).status_code
        == 404
    )
    assert client.patch("/sessions/9999/acceptance-criteria/1", json=body).status_code == 404


def test_update_status_toggle(client: TestClient) -> None:
    session = _create_session(client)
    items = _generate(client, session["id"])
    target = items[0]

    accepted = client.patch(
        f"/sessions/{session['id']}/acceptance-criteria/{target['id']}/status",
        json={"status": "accepted"},
    )
    assert accepted.status_code == 200
    assert accepted.json()["status"] == "accepted"

    back_to_pending = client.patch(
        f"/sessions/{session['id']}/acceptance-criteria/{target['id']}/status",
        json={"status": "pending"},
    )
    assert back_to_pending.json()["status"] == "pending"


def test_update_status_404_for_missing_ac_or_session(client: TestClient) -> None:
    session = _create_session(client)
    _generate(client, session["id"])
    body = {"status": "rejected"}

    assert (
        client.patch(
            f"/sessions/{session['id']}/acceptance-criteria/9999/status", json=body
        ).status_code
        == 404
    )
    assert client.patch("/sessions/9999/acceptance-criteria/1/status", json=body).status_code == 404


def test_regenerate_selected_returns_candidates_without_mutating_persisted_list(
    client: TestClient,
) -> None:
    session = _create_session(client)
    items = _generate(client, session["id"])
    target = items[0]
    before = client.get(f"/sessions/{session['id']}/acceptance-criteria").json()

    response = client.post(
        f"/sessions/{session['id']}/acceptance-criteria/{target['id']}/regenerate",
        json={"messages": [{"role": "user", "text": "more context please", "attachments": []}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["reply"]["role"] == "assistant"
    assert isinstance(body["reply"]["text"], str) and body["reply"]["text"]
    assert 1 <= len(body["candidates"]) <= 2

    after = client.get(f"/sessions/{session['id']}/acceptance-criteria").json()
    assert after == before


def test_regenerate_selected_404_for_missing_ac(client: TestClient) -> None:
    session = _create_session(client)
    response = client.post(
        f"/sessions/{session['id']}/acceptance-criteria/9999/regenerate",
        json={"messages": []},
    )
    assert response.status_code == 404


def test_apply_approved_replaces_target_and_leaves_rest_untouched(client: TestClient) -> None:
    session = _create_session(client)
    items = _generate(client, session["id"])
    target = items[1]
    others = [item for item in items if item["id"] != target["id"]]

    regenerate_response = client.post(
        f"/sessions/{session['id']}/acceptance-criteria/{target['id']}/regenerate",
        json={"messages": []},
    )
    candidates = regenerate_response.json()["candidates"]

    apply_response = client.post(
        f"/sessions/{session['id']}/acceptance-criteria/{target['id']}/apply-approved",
        json={"candidates": candidates},
    )
    assert apply_response.status_code == 200
    updated_list = apply_response.json()["acceptance_criteria"]

    updated_ids = {item["id"] for item in updated_list}
    assert target["id"] not in updated_ids
    for other in others:
        assert other["id"] in updated_ids
    assert len(updated_list) == len(others) + len(candidates)


def test_apply_approved_400_when_no_candidates(client: TestClient) -> None:
    session = _create_session(client)
    items = _generate(client, session["id"])
    response = client.post(
        f"/sessions/{session['id']}/acceptance-criteria/{items[0]['id']}/apply-approved",
        json={"candidates": []},
    )
    assert response.status_code == 400


def test_apply_approved_404_for_missing_ac(client: TestClient) -> None:
    session = _create_session(client)
    _generate(client, session["id"])
    response = client.post(
        f"/sessions/{session['id']}/acceptance-criteria/9999/apply-approved",
        json={"candidates": [{"id": -1, "title": "t", "given": "g", "when": "w", "then": "t", "scores": {"relevance": 5, "correctness": 5, "understandability": 5, "coverage": 5}, "overall_score": 5}]},
    )
    assert response.status_code == 404


def test_regenerate_all_kickoff_message_references_rejected_titles(client: TestClient) -> None:
    session = _create_session(client)
    items = _generate(client, session["id"])
    rejected = items[0]
    client.patch(
        f"/sessions/{session['id']}/acceptance-criteria/{rejected['id']}/status",
        json={"status": "rejected"},
    )

    response = client.post(f"/sessions/{session['id']}/acceptance-criteria/regenerate-all-kickoff")
    assert response.status_code == 201
    body = response.json()
    assert body["role"] == "assistant"
    assert rejected["title"] in body["text"]

    detail = client.get(f"/sessions/{session['id']}").json()
    assert len(detail["messages"]) == 1
    assert detail["messages"][0]["role"] == "assistant"
    assert detail["updated_at"] >= session["updated_at"]


def test_regenerate_all_kickoff_404_for_missing_session(client: TestClient) -> None:
    response = client.post("/sessions/9999/acceptance-criteria/regenerate-all-kickoff")
    assert response.status_code == 404


def test_delete_session_with_acceptance_criteria(client: TestClient) -> None:
    session = _create_session(client)
    _generate(client, session["id"])

    response = client.delete(f"/sessions/{session['id']}")
    assert response.status_code == 204
    assert client.get(f"/sessions/{session['id']}/acceptance-criteria").status_code == 404
