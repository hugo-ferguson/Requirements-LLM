from fastapi.testclient import TestClient


def _create_session(client: TestClient) -> dict:
    response = client.post("/sessions", json={})
    assert response.status_code == 201
    return response.json()


def _generate_ac(client: TestClient, session_id: int) -> list[dict]:
    response = client.post(f"/sessions/{session_id}/generate")
    assert response.status_code == 200
    return response.json()["acceptance_criteria"]


def _accept(client: TestClient, session_id: int, ac_id: int) -> None:
    response = client.patch(
        f"/sessions/{session_id}/acceptance-criteria/{ac_id}/status",
        json={"status": "accepted"},
    )
    assert response.status_code == 200


def _generate_uat(client: TestClient, session_id: int) -> list[dict]:
    response = client.post(f"/sessions/{session_id}/uat-cases/generate")
    assert response.status_code == 200
    return response.json()["groups"]


def test_generate_persists_cases_grouped_by_accepted_ac_only(client: TestClient) -> None:
    session = _create_session(client)
    ac_items = _generate_ac(client, session["id"])
    accepted_ids = {ac_items[0]["id"], ac_items[1]["id"]}
    for ac_id in accepted_ids:
        _accept(client, session["id"], ac_id)

    groups = _generate_uat(client, session["id"])
    assert len(groups) == len(accepted_ids)
    for group in groups:
        assert group["ac"]["id"] in accepted_ids
        assert len(group["uat_cases"]) >= 1
        assert all(c["status"] == "pending" for c in group["uat_cases"])


def test_generate_ignores_non_accepted_acs(client: TestClient) -> None:
    session = _create_session(client)
    ac_items = _generate_ac(client, session["id"])
    _accept(client, session["id"], ac_items[0]["id"])
    non_accepted_id = ac_items[1]["id"]

    groups = _generate_uat(client, session["id"])
    group_ac_ids = {g["ac"]["id"] for g in groups}
    assert non_accepted_id not in group_ac_ids


def test_generate_with_zero_accepted_acs_persists_empty_groups(client: TestClient) -> None:
    session = _create_session(client)
    _generate_ac(client, session["id"])

    groups = _generate_uat(client, session["id"])
    assert groups == []


def test_generate_replaces_existing_batch(client: TestClient) -> None:
    session = _create_session(client)
    ac_items = _generate_ac(client, session["id"])
    _accept(client, session["id"], ac_items[0]["id"])
    _generate_uat(client, session["id"])
    second = _generate_uat(client, session["id"])

    listed = client.get(f"/sessions/{session['id']}/uat-cases").json()["groups"]
    assert listed == second


def test_generate_404_for_missing_session(client: TestClient) -> None:
    response = client.post("/sessions/9999/uat-cases/generate")
    assert response.status_code == 404


def test_list_uat_cases_404_for_missing_session(client: TestClient) -> None:
    response = client.get("/sessions/9999/uat-cases")
    assert response.status_code == 404


def test_list_uat_cases_survives_ac_status_change_after_generate(client: TestClient) -> None:
    session = _create_session(client)
    ac_items = _generate_ac(client, session["id"])
    ac_id = ac_items[0]["id"]
    _accept(client, session["id"], ac_id)
    _generate_uat(client, session["id"])

    # Un-accept the AC after UAT cases were generated for it.
    response = client.patch(
        f"/sessions/{session['id']}/acceptance-criteria/{ac_id}/status",
        json={"status": "pending"},
    )
    assert response.status_code == 200

    listed = client.get(f"/sessions/{session['id']}/uat-cases").json()["groups"]
    assert any(g["ac"]["id"] == ac_id for g in listed)


def test_generate_ac_clears_existing_uat_cases(client: TestClient) -> None:
    session = _create_session(client)
    ac_items = _generate_ac(client, session["id"])
    _accept(client, session["id"], ac_items[0]["id"])
    _generate_uat(client, session["id"])

    # Regenerating ACs replaces their ids, so previously generated UAT cases
    # (tied to the old ids) must be cleared, not left orphaned.
    _generate_ac(client, session["id"])
    listed = client.get(f"/sessions/{session['id']}/uat-cases").json()["groups"]
    assert listed == []


def _first_uat_id(groups: list[dict]) -> tuple[int, int]:
    """Return (ac_id, uat_id) of the first case in the first group."""
    group = groups[0]
    return group["ac"]["id"], group["uat_cases"][0]["id"]


def test_update_text_saves_edit_and_leaves_scores_and_status_untouched(client: TestClient) -> None:
    session = _create_session(client)
    ac_items = _generate_ac(client, session["id"])
    _accept(client, session["id"], ac_items[0]["id"])
    groups = _generate_uat(client, session["id"])
    _, uat_id = _first_uat_id(groups)
    original = groups[0]["uat_cases"][0]

    response = client.patch(
        f"/sessions/{session['id']}/uat-cases/{uat_id}",
        json={"title": "Edited title", "description": "Edited description"},
    )
    assert response.status_code == 200
    updated = response.json()
    assert updated["title"] == "Edited title"
    assert updated["description"] == "Edited description"
    assert updated["scores"] == original["scores"]
    assert updated["status"] == original["status"]


def test_update_text_404_for_missing_uat_or_session(client: TestClient) -> None:
    session = _create_session(client)
    body = {"title": "t", "description": "d"}
    assert (
        client.patch(f"/sessions/{session['id']}/uat-cases/9999", json=body).status_code == 404
    )
    assert client.patch("/sessions/9999/uat-cases/1", json=body).status_code == 404


def test_update_status_toggle(client: TestClient) -> None:
    session = _create_session(client)
    ac_items = _generate_ac(client, session["id"])
    _accept(client, session["id"], ac_items[0]["id"])
    groups = _generate_uat(client, session["id"])
    _, uat_id = _first_uat_id(groups)

    accepted = client.patch(
        f"/sessions/{session['id']}/uat-cases/{uat_id}/status", json={"status": "accepted"}
    )
    assert accepted.status_code == 200
    assert accepted.json()["status"] == "accepted"

    back = client.patch(
        f"/sessions/{session['id']}/uat-cases/{uat_id}/status", json={"status": "pending"}
    )
    assert back.json()["status"] == "pending"


def test_update_status_404_for_missing_uat_or_session(client: TestClient) -> None:
    session = _create_session(client)
    body = {"status": "rejected"}
    assert (
        client.patch(f"/sessions/{session['id']}/uat-cases/9999/status", json=body).status_code
        == 404
    )
    assert client.patch("/sessions/9999/uat-cases/1/status", json=body).status_code == 404


def test_regenerate_selected_returns_candidates_without_mutating_persisted_data(
    client: TestClient,
) -> None:
    session = _create_session(client)
    ac_items = _generate_ac(client, session["id"])
    _accept(client, session["id"], ac_items[0]["id"])
    groups = _generate_uat(client, session["id"])
    _, uat_id = _first_uat_id(groups)
    before = client.get(f"/sessions/{session['id']}/uat-cases").json()

    response = client.post(
        f"/sessions/{session['id']}/uat-cases/{uat_id}/regenerate",
        json={"messages": [{"role": "user", "text": "more detail please", "attachments": []}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["reply"]["role"] == "assistant"
    assert 1 <= len(body["candidates"]) <= 2

    after = client.get(f"/sessions/{session['id']}/uat-cases").json()
    assert after == before


def test_regenerate_selected_404_for_missing_uat(client: TestClient) -> None:
    session = _create_session(client)
    response = client.post(
        f"/sessions/{session['id']}/uat-cases/9999/regenerate", json={"messages": []}
    )
    assert response.status_code == 404


def test_apply_approved_merges_in_place_within_correct_ac_group(client: TestClient) -> None:
    session = _create_session(client)
    ac_items = _generate_ac(client, session["id"])
    _accept(client, session["id"], ac_items[0]["id"])
    _accept(client, session["id"], ac_items[1]["id"])
    groups = _generate_uat(client, session["id"])

    group_a = next(g for g in groups if g["ac"]["id"] == ac_items[0]["id"])
    group_b = next(g for g in groups if g["ac"]["id"] == ac_items[1]["id"])
    target = group_a["uat_cases"][0]
    other_cases_in_a = [c for c in group_a["uat_cases"] if c["id"] != target["id"]]

    regen = client.post(
        f"/sessions/{session['id']}/uat-cases/{target['id']}/regenerate", json={"messages": []}
    )
    candidates = regen.json()["candidates"]

    apply_response = client.post(
        f"/sessions/{session['id']}/uat-cases/{target['id']}/apply-approved",
        json={"candidates": candidates},
    )
    assert apply_response.status_code == 200
    updated_group = apply_response.json()
    assert updated_group["ac"]["id"] == ac_items[0]["id"]
    updated_ids = {c["id"] for c in updated_group["uat_cases"]}
    assert target["id"] not in updated_ids
    for other in other_cases_in_a:
        assert other["id"] in updated_ids
    assert len(updated_group["uat_cases"]) == len(other_cases_in_a) + len(candidates)

    # Group B (a different AC) must be completely unaffected.
    listed = client.get(f"/sessions/{session['id']}/uat-cases").json()["groups"]
    listed_group_b = next(g for g in listed if g["ac"]["id"] == ac_items[1]["id"])
    assert listed_group_b["uat_cases"] == group_b["uat_cases"]


def test_apply_approved_400_when_no_candidates(client: TestClient) -> None:
    session = _create_session(client)
    ac_items = _generate_ac(client, session["id"])
    _accept(client, session["id"], ac_items[0]["id"])
    groups = _generate_uat(client, session["id"])
    _, uat_id = _first_uat_id(groups)

    response = client.post(
        f"/sessions/{session['id']}/uat-cases/{uat_id}/apply-approved", json={"candidates": []}
    )
    assert response.status_code == 400


def test_apply_approved_404_for_missing_uat(client: TestClient) -> None:
    session = _create_session(client)
    candidate = {
        "id": -1,
        "ac_id": 1,
        "title": "t",
        "description": "d",
        "scores": {"relevance": 5, "correctness": 5, "understandability": 5, "coverage": 5},
        "overall_score": 5,
    }
    response = client.post(
        f"/sessions/{session['id']}/uat-cases/9999/apply-approved",
        json={"candidates": [candidate]},
    )
    assert response.status_code == 404


def test_delete_session_with_uat_cases(client: TestClient) -> None:
    session = _create_session(client)
    ac_items = _generate_ac(client, session["id"])
    _accept(client, session["id"], ac_items[0]["id"])
    _generate_uat(client, session["id"])

    response = client.delete(f"/sessions/{session['id']}")
    assert response.status_code == 204
    assert client.get(f"/sessions/{session['id']}/uat-cases").status_code == 404
