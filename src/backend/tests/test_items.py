from fastapi.testclient import TestClient


def test_create_and_list_items(client: TestClient) -> None:
    response = client.post("/items", json={"name": "First item"})
    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "First item"
    assert "id" in body

    response = client.get("/items")
    assert response.status_code == 200
    items = response.json()
    assert len(items) == 1
    assert items[0]["name"] == "First item"
