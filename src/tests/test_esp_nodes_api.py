"""
Tests for ESP Nodes API endpoints
"""

import pytest


class TestESPNodesAPI:
    """Test cases for ESP Nodes CRUD operations."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get('/health')
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'healthy'
    
    def test_list_empty_nodes(self, client):
        """Test listing ESP nodes when none exist."""
        response = client.get('/api/esp-nodes')
        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, list)
        assert len(data) == 0
    
    def test_create_node_success(self, create_esp_node):
        """Test creating a new ESP node."""
        node, status = create_esp_node("192.168.1.100", "Living Room")
        
        assert status == 201
        assert node['id'] == 1
        assert node['ip_address'] == "192.168.1.100"
        assert node['room_name'] == "Living Room"
        assert 'created_at' in node
    
    def test_create_multiple_nodes(self, create_esp_node, client):
        """Test creating multiple ESP nodes."""
        node1, status1 = create_esp_node("192.168.1.100", "Room 1")
        node2, status2 = create_esp_node("192.168.1.101", "Room 2")
        
        assert status1 == 201
        assert status2 == 201
        assert node1['id'] == 1
        assert node2['id'] == 2
        
        # Verify both are in list
        response = client.get('/api/esp-nodes')
        assert response.status_code == 200
        nodes = response.get_json()
        assert len(nodes) == 2
    
    def test_create_node_missing_ip(self, client):
        """Test creating node without IP address."""
        response = client.post('/api/esp-nodes', json={
            "room_name": "Living Room"
        })
        assert response.status_code == 400
        assert 'ip_address' in response.get_json()['error']
    
    def test_create_node_duplicate_ip(self, create_esp_node):
        """Test creating node with duplicate IP."""
        create_esp_node("192.168.1.100", "Room 1")
        node, status = create_esp_node("192.168.1.100", "Room 2")
        
        assert status == 409
        assert 'already exists' in node['error']
    
    def test_get_node_success(self, sample_esp_node, client):
        """Test getting a specific node."""
        node_id = sample_esp_node['id']
        response = client.get(f'/api/esp-nodes/{node_id}')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['id'] == node_id
        assert data['ip_address'] == "192.168.1.100"
    
    def test_get_node_not_found(self, client):
        """Test getting non-existent node."""
        response = client.get('/api/esp-nodes/9999')
        assert response.status_code == 404
        assert 'not found' in response.get_json()['error']
    
    def test_update_node_success(self, sample_esp_node, client):
        """Test updating a node."""
        node_id = sample_esp_node['id']
        response = client.put(f'/api/esp-nodes/{node_id}', json={
            "room_name": "Master Bedroom"
        })
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['room_name'] == "Master Bedroom"
        assert data['ip_address'] == "192.168.1.100"  # unchanged
    
    def test_update_node_not_found(self, client):
        """Test updating non-existent node."""
        response = client.put('/api/esp-nodes/9999', json={
            "room_name": "New Room"
        })
        assert response.status_code == 404
    
    def test_update_node_no_data(self, sample_esp_node, client):
        """Test updating node with no data."""
        node_id = sample_esp_node['id']
        # Flask returns 415 for no Content-Type/JSON data
        response = client.put(f'/api/esp-nodes/{node_id}')
        
        # Expect 415 for no JSON data (or 400 if we have JSON but no fields)
        assert response.status_code in [400, 415]
    
    def test_delete_node_success(self, sample_esp_node, client):
        """Test deleting a node."""
        node_id = sample_esp_node['id']
        response = client.delete(f'/api/esp-nodes/{node_id}')
        
        assert response.status_code == 200
        assert 'deleted successfully' in response.get_json()['message']
        
        # Verify deleted
        response = client.get(f'/api/esp-nodes/{node_id}')
        assert response.status_code == 404
    
    def test_delete_node_not_found(self, client):
        """Test deleting non-existent node."""
        response = client.delete('/api/esp-nodes/9999')
        assert response.status_code == 404
