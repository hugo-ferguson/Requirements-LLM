"""API tests for generating and reviewing UAT test cases."""

from fastapi.testclient import TestClient


def _create_session(client: TestClient) -> dict:
    """Create an empty session and return it."""
    response = client.post("/sessions", json={})
    assert response.status_code == 201
    return response.json()


def _generate_criteria(client: TestClient, session_id: int) -> list[dict]:
    """Post a user story, then generate and return the session's ACs.

    Generation refuses an empty conversation (see NoUserStoryError).
    """
    client.post(
        f"/sessions/{session_id}/messages",
        json={
            "text": "As a user, I want to log in with my email and password.",
            "attachments": [],
        },
    )
    response = client.post(f"/sessions/{session_id}/generate")
    assert response.status_code == 200
    return response.json()["acceptance_criteria"]


def _accept(client: TestClient, session_id: int, ac_id: int) -> None:
    """Accept an AC."""
    response = client.patch(
        f"/sessions/{session_id}/acceptance-criteria/{ac_id}/status",
        json={"status": "accepted"},
    )
    assert response.status_code == 200


def _generate_uat_cases(client: TestClient, session_id: int) -> list[dict]:
    """Generate UAT cases and return the session's groups."""
    response = client.post(f"/sessions/{session_id}/uat-cases/generate")
    assert response.status_code == 200
    return response.json()["groups"]


def test_generate_persists_cases_grouped_by_accepted_ac_only(
    client: TestClient,
) -> None:
    session = _create_session(client)
    criteria = _generate_criteria(client, session["id"])
    accepted_ids = {criteria[0]["id"], criteria[1]["id"]}
    for ac_id in accepted_ids:
        _accept(client, session["id"], ac_id)

    groups = _generate_uat_cases(client, session["id"])
    assert len(groups) == len(accepted_ids)
    for group in groups:
        assert group["ac"]["id"] in accepted_ids
        assert 2 <= len(group["uat_cases"]) <= 4
        assert all(case["status"] == "pending" for case in group["uat_cases"])


def test_generate_ignores_non_accepted_acs(client: TestClient) -> None:
    session = _create_session(client)
    criteria = _generate_criteria(client, session["id"])
    _accept(client, session["id"], criteria[0]["id"])
    non_accepted_id = criteria[1]["id"]

    groups = _generate_uat_cases(client, session["id"])
    group_ac_ids = {group["ac"]["id"] for group in groups}
    assert non_accepted_id not in group_ac_ids


def test_generate_with_zero_accepted_acs_persists_empty_groups(
    client: TestClient,
) -> None:
    session = _create_session(client)
    _generate_criteria(client, session["id"])

    groups = _generate_uat_cases(client, session["id"])
    assert groups == []


def test_generate_replaces_existing_batch(client: TestClient) -> None:
    session = _create_session(client)
    criteria = _generate_criteria(client, session["id"])
    _accept(client, session["id"], criteria[0]["id"])
    _generate_uat_cases(client, session["id"])
    second = _generate_uat_cases(client, session["id"])

    listed = client.get(f"/sessions/{session['id']}/uat-cases").json()["groups"]
    assert listed == second


def test_generate_404_for_missing_session(client: TestClient) -> None:
    response = client.post("/sessions/9999/uat-cases/generate")
    assert response.status_code == 404


def test_list_uat_cases_404_for_missing_session(client: TestClient) -> None:
    response = client.get("/sessions/9999/uat-cases")
    assert response.status_code == 404


def test_list_uat_cases_survives_ac_status_change_after_generate(
    client: TestClient,
) -> None:
    session = _create_session(client)
    criteria = _generate_criteria(client, session["id"])
    ac_id = criteria[0]["id"]
    _accept(client, session["id"], ac_id)
    _generate_uat_cases(client, session["id"])

    # Move the AC back to pending after its UAT cases were generated.
    response = client.patch(
        f"/sessions/{session['id']}/acceptance-criteria/{ac_id}/status",
        json={"status": "pending"},
    )
    assert response.status_code == 200

    listed = client.get(f"/sessions/{session['id']}/uat-cases").json()["groups"]
    assert any(group["ac"]["id"] == ac_id for group in listed)


def test_generate_ac_clears_existing_uat_cases(client: TestClient) -> None:
    session = _create_session(client)
    criteria = _generate_criteria(client, session["id"])
    _accept(client, session["id"], criteria[0]["id"])
    _generate_uat_cases(client, session["id"])

    # Regenerating ACs deletes the old ACs' UAT cases.
    _generate_criteria(client, session["id"])
    listed = client.get(f"/sessions/{session['id']}/uat-cases").json()["groups"]
    assert listed == []


def _first_uat_id(groups: list[dict]) -> tuple[int, int]:
    """Return (ac_id, uat_id) of the first case in the first group."""
    group = groups[0]
    return group["ac"]["id"], group["uat_cases"][0]["id"]


def test_update_text_saves_edit_and_leaves_scores_and_status_untouched(
    client: TestClient,
) -> None:
    session = _create_session(client)
    criteria = _generate_criteria(client, session["id"])
    _accept(client, session["id"], criteria[0]["id"])
    groups = _generate_uat_cases(client, session["id"])
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
        client.patch(f"/sessions/{session['id']}/uat-cases/9999", json=body).status_code
        == 404
    )
    assert client.patch("/sessions/9999/uat-cases/1", json=body).status_code == 404


def test_update_status_toggle(client: TestClient) -> None:
    session = _create_session(client)
    criteria = _generate_criteria(client, session["id"])
    _accept(client, session["id"], criteria[0]["id"])
    groups = _generate_uat_cases(client, session["id"])
    _, uat_id = _first_uat_id(groups)

    accepted = client.patch(
        f"/sessions/{session['id']}/uat-cases/{uat_id}/status",
        json={"status": "accepted"},
    )
    assert accepted.status_code == 200
    assert accepted.json()["status"] == "accepted"

    back = client.patch(
        f"/sessions/{session['id']}/uat-cases/{uat_id}/status",
        json={"status": "pending"},
    )
    assert back.json()["status"] == "pending"


def test_update_status_404_for_missing_uat_or_session(client: TestClient) -> None:
    session = _create_session(client)
    body = {"status": "rejected"}
    assert (
        client.patch(
            f"/sessions/{session['id']}/uat-cases/9999/status", json=body
        ).status_code
        == 404
    )
    assert (
        client.patch("/sessions/9999/uat-cases/1/status", json=body).status_code == 404
    )


def test_regenerate_selected_returns_candidates_without_mutating_persisted_data(
    client: TestClient,
) -> None:
    session = _create_session(client)
    criteria = _generate_criteria(client, session["id"])
    _accept(client, session["id"], criteria[0]["id"])
    groups = _generate_uat_cases(client, session["id"])
    _, uat_id = _first_uat_id(groups)
    before = client.get(f"/sessions/{session['id']}/uat-cases").json()

    response = client.post(
        f"/sessions/{session['id']}/uat-cases/{uat_id}/regenerate",
        json={
            "messages": [
                {"role": "user", "text": "more detail please", "attachments": []}
            ]
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["reply"]["role"] == "assistant"
    # Regeneration refines the selected case into exactly one candidate.
    assert len(body["candidates"]) == 1

    after = client.get(f"/sessions/{session['id']}/uat-cases").json()
    assert after == before


def test_regenerate_selected_404_for_missing_uat(client: TestClient) -> None:
    session = _create_session(client)
    response = client.post(
        f"/sessions/{session['id']}/uat-cases/9999/regenerate", json={"messages": []}
    )
    assert response.status_code == 404


def test_apply_approved_merges_in_place_within_correct_ac_group(
    client: TestClient,
) -> None:
    session = _create_session(client)
    criteria = _generate_criteria(client, session["id"])
    _accept(client, session["id"], criteria[0]["id"])
    _accept(client, session["id"], criteria[1]["id"])
    groups = _generate_uat_cases(client, session["id"])

    group_a = next(group for group in groups if group["ac"]["id"] == criteria[0]["id"])
    group_b = next(group for group in groups if group["ac"]["id"] == criteria[1]["id"])
    target = group_a["uat_cases"][0]
    other_cases_in_a = [
        case for case in group_a["uat_cases"] if case["id"] != target["id"]
    ]

    regenerate_response = client.post(
        f"/sessions/{session['id']}/uat-cases/{target['id']}/regenerate",
        json={"messages": []},
    )
    candidates = regenerate_response.json()["candidates"]

    apply_response = client.post(
        f"/sessions/{session['id']}/uat-cases/{target['id']}/apply-approved",
        json={"candidates": candidates},
    )
    assert apply_response.status_code == 200
    updated_group = apply_response.json()
    assert updated_group["ac"]["id"] == criteria[0]["id"]
    updated_ids = {case["id"] for case in updated_group["uat_cases"]}
    assert target["id"] not in updated_ids
    for other in other_cases_in_a:
        assert other["id"] in updated_ids
    assert len(updated_group["uat_cases"]) == len(other_cases_in_a) + len(candidates)

    # Group B belongs to a different AC, so it is unchanged.
    listed = client.get(f"/sessions/{session['id']}/uat-cases").json()["groups"]
    listed_group_b = next(
        group for group in listed if group["ac"]["id"] == criteria[1]["id"]
    )
    assert listed_group_b["uat_cases"] == group_b["uat_cases"]


def test_apply_approved_400_when_no_candidates(client: TestClient) -> None:
    session = _create_session(client)
    criteria = _generate_criteria(client, session["id"])
    _accept(client, session["id"], criteria[0]["id"])
    groups = _generate_uat_cases(client, session["id"])
    _, uat_id = _first_uat_id(groups)

    response = client.post(
        f"/sessions/{session['id']}/uat-cases/{uat_id}/apply-approved",
        json={"candidates": []},
    )
    assert response.status_code == 400


def test_apply_approved_404_for_missing_uat(client: TestClient) -> None:
    session = _create_session(client)
    candidate = {
        "id": -1,
        "ac_id": 1,
        "title": "t",
        "description": "d",
        "scores": {
            "relevance": 5,
            "correctness": 5,
            "understandability": 5,
            "coverage": 5,
        },
        "overall_score": 5,
    }
    response = client.post(
        f"/sessions/{session['id']}/uat-cases/9999/apply-approved",
        json={"candidates": [candidate]},
    )
    assert response.status_code == 404


def test_delete_session_with_uat_cases(client: TestClient) -> None:
    session = _create_session(client)
    criteria = _generate_criteria(client, session["id"])
    _accept(client, session["id"], criteria[0]["id"])
    _generate_uat_cases(client, session["id"])

    response = client.delete(f"/sessions/{session['id']}")
    assert response.status_code == 204
    assert client.get(f"/sessions/{session['id']}/uat-cases").status_code == 404
