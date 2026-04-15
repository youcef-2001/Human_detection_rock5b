"""
Tests for Logging API endpoints
"""

import pytest


class TestLoggingAPI:
    """Test cases for Logging (Audit) CRUD operations."""
    
    def test_list_empty_logs(self, client):
        """Test listing logs when none exist."""
        response = client.get('/api/logging')
        assert response.status_code == 200
        data = response.get_json()
        assert len(data) == 0
    
    def test_create_user_log_success(self, create_log):
        """Test creating a user log entry."""
        log, status = create_log(
            log_type="user",
            action_log="User logged in",
            concerned_column="users"
        )
        
        assert status == 201
        assert log['id'] == 1
        assert log['log_type'] == "user"
        assert log['action_log'] == "User logged in"
        assert log['concerned_column'] == "users"
        assert 'created_at' in log
    
    def test_create_system_log_success(self, create_log):
        """Test creating a system log entry."""
        log, status = create_log(
            log_type="system",
            action_log="System startup",
            concerned_column="system"
        )
        
        assert status == 201
        assert log['log_type'] == "system"
    
    def test_create_log_without_column(self, client):
        """Test creating log without concerned_column (optional)."""
        response = client.post('/api/logging', json={
            "log_type": "user",
            "action_log": "Some action"
        })
        
        assert response.status_code == 201
        log = response.get_json()
        assert log['concerned_column'] is None
    
    def test_create_log_missing_type(self, client):
        """Test creating log without log_type."""
        response = client.post('/api/logging', json={
            "action_log": "Some action"
        })
        
        assert response.status_code == 400
    
    def test_create_log_missing_action(self, client):
        """Test creating log without action_log."""
        response = client.post('/api/logging', json={
            "log_type": "user"
        })
        
        assert response.status_code == 400
    
    def test_create_log_invalid_type(self, client):
        """Test creating log with invalid log_type."""
        response = client.post('/api/logging', json={
            "log_type": "invalid",
            "action_log": "Some action"
        })
        
        assert response.status_code == 400
        assert 'must be' in response.get_json()['error']
    
    def test_list_logs_multiple(self, create_log, client):
        """Test listing multiple log entries."""
        create_log("user", "User action 1")
        create_log("system", "System action 1")
        create_log("user", "User action 2")
        
        response = client.get('/api/logging')
        assert response.status_code == 200
        logs = response.get_json()
        assert len(logs) == 3
    
    def test_list_logs_filter_by_type_user(self, create_log, client):
        """Test filtering logs by type=user."""
        create_log("user", "User action 1")
        create_log("system", "System action 1")
        create_log("user", "User action 2")
        create_log("system", "System action 2")
        
        response = client.get('/api/logging?log_type=user')
        assert response.status_code == 200
        logs = response.get_json()
        assert len(logs) == 2
        for log in logs:
            assert log['log_type'] == "user"
    
    def test_list_logs_filter_by_type_system(self, create_log, client):
        """Test filtering logs by type=system."""
        create_log("user", "User action 1")
        create_log("system", "System action 1")
        create_log("user", "User action 2")
        
        response = client.get('/api/logging?log_type=system')
        logs = response.get_json()
        assert len(logs) == 1
        assert logs[0]['log_type'] == "system"
    
    def test_list_logs_pagination(self, create_log, client):
        """Test pagination of log entries."""
        # Create 15 logs
        for i in range(15):
            create_log("user" if i % 2 == 0 else "system", f"Action {i}")
        
        # Test with custom limit
        response = client.get('/api/logging?limit=5')
        assert len(response.get_json()) == 5
        
        # Test with offset
        response = client.get('/api/logging?limit=5&offset=5')
        logs = response.get_json()
        assert len(logs) == 5
    
    def test_get_log_success(self, create_log, client):
        """Test getting a specific log entry."""
        log_created, _ = create_log("user", "Specific action")
        
        response = client.get(f'/api/logging/{log_created["id"]}')
        assert response.status_code == 200
        log = response.get_json()
        assert log['id'] == log_created['id']
        assert log['action_log'] == "Specific action"
    
    def test_get_log_not_found(self, client):
        """Test getting non-existent log."""
        response = client.get('/api/logging/9999')
        assert response.status_code == 404
        assert 'not found' in response.get_json()['error']
    
    def test_update_log_success(self, create_log, client):
        """Test updating a log entry."""
        log_created, _ = create_log("user", "Original action")
        
        response = client.put(f'/api/logging/{log_created["id"]}', json={
            "action_log": "Updated action"
        })
        
        assert response.status_code == 200
        log = response.get_json()
        assert log['action_log'] == "Updated action"
    
    def test_update_log_type(self, create_log, client):
        """Test updating log type."""
        log_created, _ = create_log("user", "Some action")
        
        response = client.put(f'/api/logging/{log_created["id"]}', json={
            "log_type": "system"
        })
        
        assert response.status_code == 200
        log = response.get_json()
        assert log['log_type'] == "system"
    
    def test_update_log_not_found(self, client):
        """Test updating non-existent log."""
        response = client.put('/api/logging/9999', json={"action_log": "New"})
        assert response.status_code == 404
    
    def test_delete_log_success(self, create_log, client):
        """Test deleting a log entry."""
        log_created, _ = create_log("user", "To delete")
        
        response = client.delete(f'/api/logging/{log_created["id"]}')
        assert response.status_code == 200
        assert 'deleted successfully' in response.get_json()['message']
        
        # Verify deleted
        response = client.get(f'/api/logging/{log_created["id"]}')
        assert response.status_code == 404
    
    def test_delete_log_not_found(self, client):
        """Test deleting non-existent log."""
        response = client.delete('/api/logging/9999')
        assert response.status_code == 404
    
    def test_get_log_statistics(self, create_log, client):
        """Test getting log statistics."""
        create_log("user", "User action 1")
        create_log("user", "User action 2")
        create_log("system", "System action 1")
        create_log("user", "User action 3")
        
        response = client.get('/api/logging/stats')
        assert response.status_code == 200
        stats = response.get_json()
        assert stats['total'] == 4
        assert stats['user_logs'] == 3
        assert stats['system_logs'] == 1
    
    def test_get_log_statistics_empty(self, client):
        """Test getting statistics when no logs exist."""
        response = client.get('/api/logging/stats')
        assert response.status_code == 200
        stats = response.get_json()
        assert stats['total'] == 0
        assert stats['user_logs'] == 0
        assert stats['system_logs'] == 0
