"""
Tests for Temperature API endpoints
"""

import pytest


class TestTemperaturesAPI:
    """Test cases for Temperature CRUD operations."""
    
    def test_list_empty_temperatures(self, client):
        """Test listing temperatures when none exist."""
        response = client.get('/api/temperatures')
        assert response.status_code == 200
        data = response.get_json()
        assert len(data) == 0
    
    def test_create_temperature_success(self, sample_esp_node, create_temperature):
        """Test creating a temperature record."""
        temp, status = create_temperature(
            sample_esp_node['id'],
            event_key="sensor_001_time",
            temperature=25.5
        )
        
        assert status == 201
        assert temp['id'] == 1
        assert temp['esp_node_id'] == sample_esp_node['id']
        assert temp['event_key'] == "sensor_001_time"
        assert temp['temperature'] == 25.5
        assert 'measured_at' in temp
        assert 'created_at' in temp
    
    def test_create_temperature_invalid_node(self, create_temperature):
        """Test creating temperature for non-existent node."""
        temp, status = create_temperature(
            esp_node_id=9999,
            event_key="sensor_invalid",
            temperature=25.0
        )
        
        assert status == 404
        assert 'not found' in temp['error']
    
    def test_create_temperature_missing_field(self, sample_esp_node, client):
        """Test creating temperature with missing required field."""
        response = client.post('/api/temperatures', json={
            "esp_node_id": sample_esp_node['id'],
            "event_key": "sensor_001",
            # missing temperature and measured_at
        })
        
        assert response.status_code == 400
    
    def test_create_temperature_duplicate_event_key(self, sample_esp_node, create_temperature):
        """Test creating temperature with duplicate event_key."""
        create_temperature(sample_esp_node['id'], event_key="unique_key_1", temperature=25.0)
        temp, status = create_temperature(sample_esp_node['id'], event_key="unique_key_1", temperature=26.0)
        
        assert status == 409
        assert 'already exists' in temp['error']
    
    def test_list_temperatures_multiple(self, sample_esp_node, create_temperature, client):
        """Test listing multiple temperature records."""
        create_temperature(sample_esp_node['id'], event_key="sensor_001", temperature=25.0)
        create_temperature(sample_esp_node['id'], event_key="sensor_002", temperature=26.0)
        create_temperature(sample_esp_node['id'], event_key="sensor_003", temperature=27.0)
        
        response = client.get('/api/temperatures')
        assert response.status_code == 200
        temps = response.get_json()
        assert len(temps) == 3
    
    def test_list_temperatures_filter_by_node(self, client, create_esp_node, create_temperature):
        """Test filtering temperatures by ESP node."""
        node1, _ = create_esp_node("192.168.1.100", "Room 1")
        node2, _ = create_esp_node("192.168.1.101", "Room 2")
        
        create_temperature(node1['id'], event_key="sensor_001", temperature=25.0)
        create_temperature(node1['id'], event_key="sensor_002", temperature=25.5)
        create_temperature(node2['id'], event_key="sensor_003", temperature=26.0)
        
        # Get temperatures for node 1
        response = client.get(f'/api/temperatures?esp_node_id={node1["id"]}')
        assert response.status_code == 200
        temps = response.get_json()
        assert len(temps) == 2
        for temp in temps:
            assert temp['esp_node_id'] == node1['id']
    
    def test_list_temperatures_pagination(self, sample_esp_node, create_temperature, client):
        """Test pagination of temperature records."""
        # Create 15 records
        for i in range(15):
            create_temperature(sample_esp_node['id'], event_key=f"sensor_{i}", temperature=20.0 + i)
        
        # Test default limit (100)
        response = client.get('/api/temperatures')
        assert len(response.get_json()) == 15
        
        # Test with custom limit
        response = client.get('/api/temperatures?limit=5')
        assert len(response.get_json()) == 5
        
        # Test with offset
        response = client.get('/api/temperatures?limit=5&offset=5')
        temps = response.get_json()
        assert len(temps) == 5
    
    def test_get_temperature_success(self, sample_esp_node, create_temperature, client):
        """Test getting a specific temperature record."""
        temp_created, _ = create_temperature(
            sample_esp_node['id'],
            event_key="sensor_specific",
            temperature=25.5
        )
        
        response = client.get(f'/api/temperatures/{temp_created["id"]}')
        assert response.status_code == 200
        temp = response.get_json()
        assert temp['id'] == temp_created['id']
        assert temp['temperature'] == 25.5
    
    def test_get_temperature_not_found(self, client):
        """Test getting non-existent temperature."""
        response = client.get('/api/temperatures/9999')
        assert response.status_code == 404
        assert 'not found' in response.get_json()['error']
    
    def test_update_temperature_success(self, sample_esp_node, create_temperature, client):
        """Test updating a temperature record."""
        temp_created, _ = create_temperature(
            sample_esp_node['id'],
            event_key="sensor_update",
            temperature=25.0
        )
        
        response = client.put(f'/api/temperatures/{temp_created["id"]}', json={
            "temperature": 27.5
        })
        
        assert response.status_code == 200
        temp = response.get_json()
        assert temp['temperature'] == 27.5
    
    def test_update_temperature_with_new_time(self, sample_esp_node, create_temperature, client):
        """Test updating temperature with new measured_at."""
        temp_created, _ = create_temperature(
            sample_esp_node['id'],
            event_key="sensor_time",
            temperature=25.0,
            measured_at="2024-04-15T10:00:00"
        )
        
        response = client.put(f'/api/temperatures/{temp_created["id"]}', json={
            "measured_at": "2024-04-15T11:00:00"
        })
        
        assert response.status_code == 200
        temp = response.get_json()
        assert "11:00:00" in temp['measured_at']
    
    def test_update_temperature_not_found(self, client):
        """Test updating non-existent temperature."""
        response = client.put('/api/temperatures/9999', json={"temperature": 20.0})
        assert response.status_code == 404
    
    def test_delete_temperature_success(self, sample_esp_node, create_temperature, client):
        """Test deleting a temperature record."""
        temp_created, _ = create_temperature(
            sample_esp_node['id'],
            event_key="sensor_delete",
            temperature=25.0
        )
        
        response = client.delete(f'/api/temperatures/{temp_created["id"]}')
        assert response.status_code == 200
        assert 'deleted successfully' in response.get_json()['message']
        
        # Verify deleted
        response = client.get(f'/api/temperatures/{temp_created["id"]}')
        assert response.status_code == 404
    
    def test_delete_temperature_not_found(self, client):
        """Test deleting non-existent temperature."""
        response = client.delete('/api/temperatures/9999')
        assert response.status_code == 404
