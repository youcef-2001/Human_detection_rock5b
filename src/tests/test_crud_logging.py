"""Tests for CRUD operational logging."""

import logging


def test_scenario_crud_logs(client, caplog):
    """Scenario create/update/delete should emit info logs."""
    caplog.set_level(logging.INFO)

    create_response = client.post(
        "/api/scenarios",
        json={
            "name": "Journal Test",
            "description": "desc",
            "is_active": True,
            "esp_node_ids": [],
        },
    )
    assert create_response.status_code == 201
    scenario_id = create_response.get_json()["id"]

    update_response = client.put(
        f"/api/scenarios/{scenario_id}",
        json={"description": "updated", "is_active": False},
    )
    assert update_response.status_code == 200

    delete_response = client.delete(f"/api/scenarios/{scenario_id}")
    assert delete_response.status_code == 200

    assert "Created scenario id=" in caplog.text
    assert "Updated scenario id=" in caplog.text
    assert "Deleted scenario id=" in caplog.text


def test_esp_node_crud_logs(client, caplog):
    """ESP node create/update/delete should emit info logs."""
    caplog.set_level(logging.INFO)

    create_response = client.post(
        "/api/esp-nodes",
        json={"ip_address": "10.20.30.40", "room_name": "Logs"},
    )
    assert create_response.status_code == 201
    node_id = create_response.get_json()["id"]

    update_response = client.put(
        f"/api/esp-nodes/{node_id}",
        json={"room_name": "Logs-updated"},
    )
    assert update_response.status_code == 200

    delete_response = client.delete(f"/api/esp-nodes/{node_id}")
    assert delete_response.status_code == 200

    assert "Created ESP node id=" in caplog.text
    assert "Updated ESP node id=" in caplog.text
    assert "Deleted ESP node id=" in caplog.text
