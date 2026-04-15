"""
Tests for Scenario API endpoints
"""

import pytest


class TestScenariosAPI:
    """Test cases for Scenario CRUD operations."""
    
    def test_list_empty_scenarios(self, client):
        """Test listing scenarios when none exist."""
        response = client.get('/api/scenarios')
        assert response.status_code == 200
        data = response.get_json()
        assert len(data) == 0
    
    def test_create_scenario_success(self, create_scenario):
        """Test creating a scenario."""
        scenario, status = create_scenario(
            name="Living Room Detection",
            description="Detect humans in living room",
            is_active=True,
            esp_node_ids=[]
        )
        
        assert status == 201
        assert scenario['id'] == 1
        assert scenario['name'] == "Living Room Detection"
        assert scenario['description'] == "Detect humans in living room"
        assert scenario['is_active'] is True
        assert scenario['esp_nodes'] == []
    
    def test_create_scenario_with_nodes(self, create_esp_node, create_scenario):
        """Test creating scenario with associated ESP nodes."""
        node1, _ = create_esp_node("192.168.1.100", "Room 1")
        node2, _ = create_esp_node("192.168.1.101", "Room 2")
        
        scenario, status = create_scenario(
            name="Multi Room",
            esp_node_ids=[node1['id'], node2['id']]
        )
        
        assert status == 201
        assert len(scenario['esp_nodes']) == 2
        node_ids = [n['id'] for n in scenario['esp_nodes']]
        assert node1['id'] in node_ids
        assert node2['id'] in node_ids
    
    def test_create_scenario_missing_name(self, client):
        """Test creating scenario without name."""
        response = client.post('/api/scenarios', json={
            "description": "Test scenario"
        })
        
        assert response.status_code == 400
        assert 'name' in response.get_json()['error']
    
    def test_create_scenario_duplicate_name(self, create_scenario):
        """Test creating scenario with duplicate name."""
        create_scenario(name="Duplicate Name")
        scenario, status = create_scenario(name="Duplicate Name")
        
        assert status == 409
        assert 'already exists' in scenario['error']
    
    def test_list_scenarios_multiple(self, create_scenario, client):
        """Test listing multiple scenarios."""
        create_scenario("Scenario 1")
        create_scenario("Scenario 2")
        create_scenario("Scenario 3")
        
        response = client.get('/api/scenarios')
        assert response.status_code == 200
        scenarios = response.get_json()
        assert len(scenarios) == 3
    
    def test_list_scenarios_filter_by_active(self, create_scenario, client):
        """Test filtering scenarios by is_active."""
        create_scenario("Active 1", is_active=True)
        create_scenario("Inactive 1", is_active=False)
        create_scenario("Active 2", is_active=True)
        
        response = client.get('/api/scenarios?is_active=true')
        scenarios = response.get_json()
        assert len(scenarios) == 2
        for scenario in scenarios:
            assert scenario['is_active'] is True
    
    def test_list_scenarios_filter_inactive(self, create_scenario, client):
        """Test filtering for inactive scenarios."""
        create_scenario("Active", is_active=True)
        create_scenario("Inactive", is_active=False)
        
        response = client.get('/api/scenarios?is_active=false')
        scenarios = response.get_json()
        assert len(scenarios) == 1
        assert scenarios[0]['is_active'] is False
    
    def test_list_scenarios_pagination(self, create_scenario, client):
        """Test pagination of scenarios."""
        for i in range(15):
            create_scenario(f"Scenario {i}")
        
        # Test with custom limit
        response = client.get('/api/scenarios?limit=5')
        assert len(response.get_json()) == 5
        
        # Test with offset
        response = client.get('/api/scenarios?limit=5&offset=5')
        scenarios = response.get_json()
        assert len(scenarios) == 5
    
    def test_get_scenario_success(self, create_scenario, client):
        """Test getting a specific scenario."""
        scenario_created, _ = create_scenario("Test Scenario")
        
        response = client.get(f'/api/scenarios/{scenario_created["id"]}')
        assert response.status_code == 200
        scenario = response.get_json()
        assert scenario['id'] == scenario_created['id']
        assert scenario['name'] == "Test Scenario"
    
    def test_get_scenario_not_found(self, client):
        """Test getting non-existent scenario."""
        response = client.get('/api/scenarios/9999')
        assert response.status_code == 404
        assert 'not found' in response.get_json()['error']
    
    def test_update_scenario_success(self, create_scenario, client):
        """Test updating a scenario."""
        scenario_created, _ = create_scenario("Original Name")
        
        response = client.put(f'/api/scenarios/{scenario_created["id"]}', json={
            "description": "Updated description",
            "is_active": False
        })
        
        assert response.status_code == 200
        scenario = response.get_json()
        assert scenario['description'] == "Updated description"
        assert scenario['is_active'] is False
        assert scenario['name'] == "Original Name"  # unchanged
    
    def test_update_scenario_nodes(self, create_esp_node, create_scenario, client):
        """Test updating scenario's ESP nodes."""
        node1, _ = create_esp_node("192.168.1.100", "Room 1")
        node2, _ = create_esp_node("192.168.1.101", "Room 2")
        node3, _ = create_esp_node("192.168.1.102", "Room 3")
        
        scenario_created, _ = create_scenario("Multi Room", esp_node_ids=[node1['id']])
        assert len(scenario_created['esp_nodes']) == 1
        
        # Update with different nodes
        response = client.put(f'/api/scenarios/{scenario_created["id"]}', json={
            "esp_node_ids": [node2['id'], node3['id']]
        })
        
        scenario = response.get_json()
        assert len(scenario['esp_nodes']) == 2
        node_ids = [n['id'] for n in scenario['esp_nodes']]
        assert node2['id'] in node_ids
        assert node3['id'] in node_ids
        assert node1['id'] not in node_ids
    
    def test_update_scenario_not_found(self, client):
        """Test updating non-existent scenario."""
        response = client.put('/api/scenarios/9999', json={"description": "New"})
        assert response.status_code == 404
    
    def test_delete_scenario_success(self, create_scenario, client):
        """Test deleting a scenario."""
        scenario_created, _ = create_scenario("To Delete")
        
        response = client.delete(f'/api/scenarios/{scenario_created["id"]}')
        assert response.status_code == 200
        assert 'deleted successfully' in response.get_json()['message']
        
        # Verify deleted
        response = client.get(f'/api/scenarios/{scenario_created["id"]}')
        assert response.status_code == 404
    
    def test_delete_scenario_not_found(self, client):
        """Test deleting non-existent scenario."""
        response = client.delete('/api/scenarios/9999')
        assert response.status_code == 404
    
    def test_add_esp_node_to_scenario(self, create_esp_node, create_scenario, client):
        """Test adding an ESP node to a scenario."""
        node, _ = create_esp_node("192.168.1.100", "Room 1")
        scenario_created, _ = create_scenario("Empty Scenario", esp_node_ids=[])
        
        assert len(scenario_created['esp_nodes']) == 0
        
        response = client.post(f'/api/scenarios/{scenario_created["id"]}/esp-nodes', json={
            "esp_node_id": node['id']
        })
        
        assert response.status_code == 200
        scenario = response.get_json()
        assert len(scenario['esp_nodes']) == 1
        assert scenario['esp_nodes'][0]['id'] == node['id']
    
    def test_add_node_scenario_not_found(self, create_esp_node, client):
        """Test adding node to non-existent scenario."""
        node, _ = create_esp_node("192.168.1.100", "Room")
        
        response = client.post('/api/scenarios/9999/esp-nodes', json={
            "esp_node_id": node['id']
        })
        
        assert response.status_code == 404
        assert 'Scenario' in response.get_json()['error']
    
    def test_add_nonexistent_node_to_scenario(self, create_scenario, client):
        """Test adding non-existent node to scenario."""
        scenario_created, _ = create_scenario("Test")
        
        response = client.post(f'/api/scenarios/{scenario_created["id"]}/esp-nodes', json={
            "esp_node_id": 9999
        })
        
        assert response.status_code == 404
        assert 'ESP node' in response.get_json()['error']
    
    def test_remove_esp_node_from_scenario(self, create_esp_node, create_scenario, client):
        """Test removing an ESP node from a scenario."""
        node1, _ = create_esp_node("192.168.1.100", "Room 1")
        node2, _ = create_esp_node("192.168.1.101", "Room 2")
        
        scenario_created, _ = create_scenario(
            "Two Nodes",
            esp_node_ids=[node1['id'], node2['id']]
        )
        
        assert len(scenario_created['esp_nodes']) == 2
        
        # Remove node 1
        response = client.delete(f'/api/scenarios/{scenario_created["id"]}/esp-nodes/{node1["id"]}')
        
        assert response.status_code == 200
        scenario = response.get_json()
        assert len(scenario['esp_nodes']) == 1
        assert scenario['esp_nodes'][0]['id'] == node2['id']
    
    def test_remove_node_scenario_not_found(self, client):
        """Test removing node from non-existent scenario."""
        response = client.delete('/api/scenarios/9999/esp-nodes/1')
        assert response.status_code == 404
    
    def test_remove_nonexistent_node(self, create_scenario, client):
        """Test removing non-existent node from scenario."""
        scenario_created, _ = create_scenario("Test")
        
        response = client.delete(f'/api/scenarios/{scenario_created["id"]}/esp-nodes/9999')
        assert response.status_code == 404
