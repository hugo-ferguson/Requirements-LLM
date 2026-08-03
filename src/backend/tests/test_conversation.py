from fastapi.testclient import TestClient


def test_send_message_returns_canned_assistant_reply(client: TestClient) -> None:
    response = client.post(
        "/conversation/messages",
        json={"messages": [{"role": "user", "text": "Hello", "attachments": []}]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["role"] == "assistant"
    assert isinstance(body["text"], str) and body["text"]


def test_generate_returns_example_acceptance_criteria(client: TestClient) -> None:
    response = client.post(
        "/conversation/generate",
        json={
            "messages": [
                {
                    "role": "user",
                    "text": "Attached are the user stories",
                    "attachments": [
                        {"filename": "UserStorySet1.txt", "content": "As a user..."}
                    ],
                }
            ]
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["acceptance_criteria"]) >= 1
    first = body["acceptance_criteria"][0]
    assert {"id", "title", "given", "when", "then", "scores", "overall_score"} <= first.keys()
    assert {"relevance", "correctness", "understandability", "coverage"} <= first[
        "scores"
    ].keys()


def test_generate_ignores_conversation_content(client: TestClient) -> None:
    response_a = client.post("/conversation/generate", json={"messages": []})
    response_b = client.post(
        "/conversation/generate",
        json={
            "messages": [
                {"role": "user", "text": "completely different content", "attachments": []}
            ]
        },
    )
    assert response_a.json() == response_b.json()
