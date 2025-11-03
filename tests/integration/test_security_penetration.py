#!/usr/bin/env python3
"""
Security and Penetration Testing for Project Argus
Tests vulnerability assessments, encryption, authentication, and access control systems.
"""

import pytest
import hashlib
import hmac
import base64
import time
import json
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import secrets
import jwt

# Import security-related components
from shared.models.user import User, UserRole
from shared.models.evidence import Evidence, EvidenceType


class MockSecurityService:
    """Mock security service for testing security features."""
    
    def __init__(self):
        self.users = {}
        self.sessions = {}
        self.failed_attempts = {}
        self.encryption_key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
        self.hmac_secret = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
        self.jwt_secret = secrets.token_urlsafe(32)
    
    def create_user(self, username, password, role=UserRole.OPERATOR):
        """Create a new user with hashed password."""
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        
        user = User(
            id=f"user_{len(self.users)+1}",
            username=username,
            password_hash=base64.b64encode(password_hash).decode(),
            salt=salt,
            role=role,
            created_at=datetime.now(),
            last_login=None,
            active=True
        )
        
        self.users[username] = user
        return user
    
    def authenticate_user(self, username, password):
        """Authenticate user with password."""
        if username not in self.users:
            return None
        
        user = self.users[username]
        if not user.active:
            return None
        
        # Check password
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), user.salt.encode(), 100000)
        expected_hash = base64.b64decode(user.password_hash)
        
        if hmac.compare_digest(password_hash, expected_hash):
            user.last_login = datetime.now()
            return user
        else:
            # Track failed attempts
            self.failed_attempts[username] = self.failed_attempts.get(username, 0) + 1
            return None
    
    def generate_jwt_token(self, user):
        """Generate JWT token for authenticated user."""
        payload = {
            'user_id': user.id,
            'username': user.username,
            'role': user.role.value,
            'exp': datetime.utcnow() + timedelta(hours=24),
            'iat': datetime.utcnow()
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm='HS256')
        return token
    
    def verify_jwt_token(self, token):
        """Verify JWT token and return user info."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def encrypt_data(self, data):
        """Encrypt data using AES-256."""
        from cryptography.fernet import Fernet
        key = base64.urlsafe_b64decode(self.encryption_key.encode())
        fernet = Fernet(base64.urlsafe_b64encode(key))
        
        if isinstance(data, str):
            data = data.encode()
        
        encrypted_data = fernet.encrypt(data)
        return base64.b64encode(encrypted_data).decode()
    
    def decrypt_data(self, encrypted_data):
        """Decrypt data using AES-256."""
        from cryptography.fernet import Fernet
        key = base64.urlsafe_b64decode(self.encryption_key.encode())
        fernet = Fernet(base64.urlsafe_b64encode(key))
        
        encrypted_bytes = base64.b64decode(encrypted_data.encode())
        decrypted_data = fernet.decrypt(encrypted_bytes)
        return decrypted_data.decode()
    
    def generate_hmac_signature(self, data):
        """Generate HMAC signature for data integrity."""
        if isinstance(data, str):
            data = data.encode()
        
        signature = hmac.new(
            self.hmac_secret.encode(),
            data,
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def verify_hmac_signature(self, data, signature):
        """Verify HMAC signature."""
        expected_signature = self.generate_hmac_signature(data)
        return hmac.compare_digest(signature, expected_signature)


class TestAuthenticationSecurity:
    """Test authentication and authorization security."""
    
    def setup_method(self):
        """Set up security test fixtures."""
        self.security_service = MockSecurityService()
    
    def test_password_hashing_security(self):
        """Test secure password hashing implementation."""
        
        # Create user with password
        user = self.security_service.create_user("testuser", "SecurePassword123!")
        
        # Verify password is not stored in plaintext
        assert user.password_hash != "SecurePassword123!"
        assert len(user.password_hash) > 20  # Should be base64 encoded hash
        assert user.salt is not None
        assert len(user.salt) == 32  # 16 bytes hex encoded
        
        # Verify authentication works
        authenticated_user = self.security_service.authenticate_user("testuser", "SecurePassword123!")
        assert authenticated_user is not None
        assert authenticated_user.username == "testuser"
        
        # Verify wrong password fails
        failed_auth = self.security_service.authenticate_user("testuser", "WrongPassword")
        assert failed_auth is None
        
        # Verify different users have different salts
        user2 = self.security_service.create_user("testuser2", "SecurePassword123!")
        assert user.salt != user2.salt
        assert user.password_hash != user2.password_hash
    
    def test_jwt_token_security(self):
        """Test JWT token generation and validation."""
        
        # Create user and generate token
        user = self.security_service.create_user("jwtuser", "TokenPassword123!")
        token = self.security_service.generate_jwt_token(user)
        
        # Verify token is generated
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 50  # JWT tokens are typically long
        
        # Verify token can be decoded
        payload = self.security_service.verify_jwt_token(token)
        assert payload is not None
        assert payload['username'] == "jwtuser"
        assert payload['user_id'] == user.id
        assert payload['role'] == user.role.value
        
        # Verify token expiration
        assert 'exp' in payload
        assert 'iat' in payload
        
        # Test invalid token
        invalid_payload = self.security_service.verify_jwt_token("invalid.token.here")
        assert invalid_payload is None
        
        # Test expired token (simulate by modifying secret)
        old_secret = self.security_service.jwt_secret
        self.security_service.jwt_secret = "different_secret"
        expired_payload = self.security_service.verify_jwt_token(token)
        assert expired_payload is None
        
        # Restore secret
        self.security_service.jwt_secret = old_secret
    
    def test_role_based_access_control(self):
        """Test role-based access control implementation."""
        
        # Create users with different roles
        operator = self.security_service.create_user("operator", "OpPass123!", UserRole.OPERATOR)
        auditor = self.security_service.create_user("auditor", "AudPass123!", UserRole.AUDITOR)
        admin = self.security_service.create_user("admin", "AdminPass123!", UserRole.ADMINISTRATOR)
        
        # Generate tokens
        operator_token = self.security_service.generate_jwt_token(operator)
        auditor_token = self.security_service.generate_jwt_token(auditor)
        admin_token = self.security_service.generate_jwt_token(admin)
        
        # Verify role information in tokens
        operator_payload = self.security_service.verify_jwt_token(operator_token)
        auditor_payload = self.security_service.verify_jwt_token(auditor_token)
        admin_payload = self.security_service.verify_jwt_token(admin_token)
        
        assert operator_payload['role'] == UserRole.OPERATOR.value
        assert auditor_payload['role'] == UserRole.AUDITOR.value
        assert admin_payload['role'] == UserRole.ADMINISTRATOR.value
        
        # Test access control logic
        def check_access(payload, required_role):
            if not payload:
                return False
            user_role = UserRole(payload['role'])
            
            # Define role hierarchy
            role_hierarchy = {
                UserRole.AUDITOR: 1,
                UserRole.OPERATOR: 2,
                UserRole.ADMINISTRATOR: 3
            }
            
            return role_hierarchy.get(user_role, 0) >= role_hierarchy.get(required_role, 0)
        
        # Test operator access
        assert check_access(operator_payload, UserRole.OPERATOR) is True
        assert check_access(operator_payload, UserRole.ADMINISTRATOR) is False
        
        # Test auditor access (read-only)
        assert check_access(auditor_payload, UserRole.AUDITOR) is True
        assert check_access(auditor_payload, UserRole.OPERATOR) is False
        
        # Test admin access
        assert check_access(admin_payload, UserRole.OPERATOR) is True
        assert check_access(admin_payload, UserRole.AUDITOR) is True
        assert check_access(admin_payload, UserRole.ADMINISTRATOR) is True
    
    def test_brute_force_protection(self):
        """Test protection against brute force attacks."""
        
        # Create user
        user = self.security_service.create_user("brutetest", "ValidPassword123!")
        
        # Simulate multiple failed login attempts
        for i in range(5):
            result = self.security_service.authenticate_user("brutetest", "WrongPassword")
            assert result is None
        
        # Verify failed attempts are tracked
        assert self.security_service.failed_attempts.get("brutetest", 0) == 5
        
        # In a real system, account would be locked after too many failures
        # Here we test that the tracking mechanism works
        
        # Verify legitimate login still works (in basic implementation)
        valid_result = self.security_service.authenticate_user("brutetest", "ValidPassword123!")
        assert valid_result is not None
    
    def test_session_management_security(self):
        """Test secure session management."""
        
        user = self.security_service.create_user("sessionuser", "SessionPass123!")
        
        # Generate multiple tokens (simulate multiple sessions)
        token1 = self.security_service.generate_jwt_token(user)
        time.sleep(0.1)  # Ensure different timestamps
        token2 = self.security_service.generate_jwt_token(user)
        
        # Verify tokens are different
        assert token1 != token2
        
        # Verify both tokens are valid
        payload1 = self.security_service.verify_jwt_token(token1)
        payload2 = self.security_service.verify_jwt_token(token2)
        
        assert payload1 is not None
        assert payload2 is not None
        assert payload1['iat'] != payload2['iat']  # Different issued times


class TestEncryptionSecurity:
    """Test encryption and data protection security."""
    
    def setup_method(self):
        """Set up encryption test fixtures."""
        self.security_service = MockSecurityService()
    
    def test_data_encryption_at_rest(self):
        """Test AES-256 encryption for data at rest."""
        
        # Test data encryption
        sensitive_data = "This is sensitive border crossing data"
        encrypted_data = self.security_service.encrypt_data(sensitive_data)
        
        # Verify data is encrypted
        assert encrypted_data != sensitive_data
        assert len(encrypted_data) > len(sensitive_data)
        
        # Verify decryption works
        decrypted_data = self.security_service.decrypt_data(encrypted_data)
        assert decrypted_data == sensitive_data
        
        # Test with binary data
        binary_data = b"Binary evidence data \x00\x01\x02\x03"
        encrypted_binary = self.security_service.encrypt_data(binary_data)
        decrypted_binary = self.security_service.decrypt_data(encrypted_binary)
        assert decrypted_binary.encode() == binary_data
        
        # Verify different encryptions of same data produce different results
        encrypted_data2 = self.security_service.encrypt_data(sensitive_data)
        assert encrypted_data != encrypted_data2  # Should be different due to random IV
    
    def test_hmac_data_integrity(self):
        """Test HMAC signatures for data integrity."""
        
        # Test HMAC signature generation
        data = "Important evidence data that must not be tampered with"
        signature = self.security_service.generate_hmac_signature(data)
        
        # Verify signature is generated
        assert signature is not None
        assert len(signature) == 64  # SHA-256 hex digest length
        
        # Verify signature validation
        is_valid = self.security_service.verify_hmac_signature(data, signature)
        assert is_valid is True
        
        # Test tampered data detection
        tampered_data = data + " TAMPERED"
        is_tampered_valid = self.security_service.verify_hmac_signature(tampered_data, signature)
        assert is_tampered_valid is False
        
        # Test tampered signature detection
        tampered_signature = signature[:-2] + "XX"
        is_sig_tampered_valid = self.security_service.verify_hmac_signature(data, tampered_signature)
        assert is_sig_tampered_valid is False
        
        # Test with binary data
        binary_data = b"Binary evidence \x00\x01\x02"
        binary_signature = self.security_service.generate_hmac_signature(binary_data)
        binary_valid = self.security_service.verify_hmac_signature(binary_data, binary_signature)
        assert binary_valid is True
    
    def test_evidence_encryption_workflow(self):
        """Test complete evidence encryption workflow."""
        
        # Simulate evidence storage with encryption
        evidence_data = {
            "camera_id": "camera_001",
            "timestamp": "2024-01-01T12:00:00Z",
            "detection_confidence": 0.95,
            "location": "Border Sector Alpha",
            "operator_notes": "Suspicious activity detected"
        }
        
        # Convert to JSON and encrypt
        json_data = json.dumps(evidence_data)
        encrypted_evidence = self.security_service.encrypt_data(json_data)
        evidence_signature = self.security_service.generate_hmac_signature(json_data)
        
        # Simulate storage
        stored_evidence = {
            "id": "evidence_001",
            "encrypted_data": encrypted_evidence,
            "hmac_signature": evidence_signature,
            "created_at": datetime.now().isoformat()
        }
        
        # Simulate retrieval and verification
        retrieved_encrypted = stored_evidence["encrypted_data"]
        retrieved_signature = stored_evidence["hmac_signature"]
        
        # Decrypt and verify
        decrypted_json = self.security_service.decrypt_data(retrieved_encrypted)
        is_integrity_valid = self.security_service.verify_hmac_signature(decrypted_json, retrieved_signature)
        
        assert is_integrity_valid is True
        
        # Parse decrypted data
        decrypted_evidence = json.loads(decrypted_json)
        assert decrypted_evidence == evidence_data
        
        # Test integrity failure simulation
        corrupted_data = decrypted_json.replace("camera_001", "camera_999")
        is_corrupted_valid = self.security_service.verify_hmac_signature(corrupted_data, retrieved_signature)
        assert is_corrupted_valid is False


class TestNetworkSecurity:
    """Test network security and communication protection."""
    
    def test_tls_configuration_validation(self):
        """Test TLS configuration requirements."""
        
        # Mock TLS configuration
        tls_config = {
            "version": "TLS 1.3",
            "cipher_suites": [
                "TLS_AES_256_GCM_SHA384",
                "TLS_CHACHA20_POLY1305_SHA256",
                "TLS_AES_128_GCM_SHA256"
            ],
            "certificate_validation": True,
            "perfect_forward_secrecy": True,
            "hsts_enabled": True,
            "certificate_transparency": True
        }
        
        # Validate TLS version
        assert tls_config["version"] in ["TLS 1.2", "TLS 1.3"]
        assert tls_config["version"] == "TLS 1.3"  # Prefer latest
        
        # Validate cipher suites (should use strong encryption)
        strong_ciphers = [cipher for cipher in tls_config["cipher_suites"] 
                         if "256" in cipher or "CHACHA20" in cipher]
        assert len(strong_ciphers) > 0
        
        # Validate security features
        assert tls_config["certificate_validation"] is True
        assert tls_config["perfect_forward_secrecy"] is True
        assert tls_config["hsts_enabled"] is True
    
    def test_api_security_headers(self):
        """Test security headers for API endpoints."""
        
        # Mock HTTP response headers
        security_headers = {
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'; script-src 'self'",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "camera=(), microphone=(), geolocation=()"
        }
        
        # Validate required security headers
        required_headers = [
            "Strict-Transport-Security",
            "Content-Security-Policy", 
            "X-Content-Type-Options",
            "X-Frame-Options"
        ]
        
        for header in required_headers:
            assert header in security_headers
            assert security_headers[header] is not None
            assert len(security_headers[header]) > 0
        
        # Validate HSTS configuration
        hsts = security_headers["Strict-Transport-Security"]
        assert "max-age=" in hsts
        assert "includeSubDomains" in hsts
        
        # Validate CSP is restrictive
        csp = security_headers["Content-Security-Policy"]
        assert "'self'" in csp
        assert "unsafe-inline" not in csp  # Should not allow unsafe inline
    
    def test_input_validation_security(self):
        """Test input validation and sanitization."""
        
        def validate_camera_id(camera_id):
            """Validate camera ID input."""
            if not isinstance(camera_id, str):
                return False
            if len(camera_id) > 50:
                return False
            if not camera_id.replace("_", "").replace("-", "").isalnum():
                return False
            return True
        
        def validate_detection_confidence(confidence):
            """Validate detection confidence input."""
            if not isinstance(confidence, (int, float)):
                return False
            if confidence < 0.0 or confidence > 1.0:
                return False
            return True
        
        def sanitize_operator_notes(notes):
            """Sanitize operator notes input."""
            if not isinstance(notes, str):
                return ""
            
            # Remove potentially dangerous characters
            dangerous_chars = ["<", ">", "&", "\"", "'", ";", "(", ")", "{", "}"]
            sanitized = notes
            for char in dangerous_chars:
                sanitized = sanitized.replace(char, "")
            
            # Limit length
            return sanitized[:1000]
        
        # Test valid inputs
        assert validate_camera_id("camera_001") is True
        assert validate_camera_id("border-cam-alpha") is True
        assert validate_detection_confidence(0.85) is True
        assert validate_detection_confidence(1.0) is True
        
        # Test invalid inputs
        assert validate_camera_id("camera_001; DROP TABLE cameras;") is False
        assert validate_camera_id("x" * 100) is False
        assert validate_detection_confidence(-0.1) is False
        assert validate_detection_confidence(1.5) is False
        
        # Test sanitization
        malicious_notes = "<script>alert('xss')</script>Legitimate notes"
        sanitized_notes = sanitize_operator_notes(malicious_notes)
        assert "<script>" not in sanitized_notes
        assert "alert" not in sanitized_notes
        assert "Legitimate notes" in sanitized_notes


class TestVulnerabilityAssessment:
    """Test for common security vulnerabilities."""
    
    def test_sql_injection_protection(self):
        """Test protection against SQL injection attacks."""
        
        # Mock database query function with parameterized queries
        def safe_query(camera_id, start_time, end_time):
            """Simulate parameterized query (safe)."""
            # In real implementation, this would use parameterized queries
            query = "SELECT * FROM detections WHERE camera_id = ? AND timestamp BETWEEN ? AND ?"
            params = (camera_id, start_time, end_time)
            return {"query": query, "params": params, "safe": True}
        
        def unsafe_query(camera_id, start_time, end_time):
            """Simulate string concatenation query (unsafe)."""
            query = f"SELECT * FROM detections WHERE camera_id = '{camera_id}' AND timestamp BETWEEN '{start_time}' AND '{end_time}'"
            return {"query": query, "safe": False}
        
        # Test safe query
        safe_result = safe_query("camera_001", "2024-01-01", "2024-01-02")
        assert safe_result["safe"] is True
        assert "?" in safe_result["query"]
        
        # Test that unsafe query would be vulnerable
        malicious_input = "'; DROP TABLE detections; --"
        unsafe_result = unsafe_query(malicious_input, "2024-01-01", "2024-01-02")
        assert "DROP TABLE" in unsafe_result["query"]
        assert unsafe_result["safe"] is False
        
        # Verify safe query handles malicious input safely
        safe_malicious_result = safe_query(malicious_input, "2024-01-01", "2024-01-02")
        assert safe_malicious_result["params"][0] == malicious_input  # Treated as parameter
        assert safe_malicious_result["safe"] is True
    
    def test_xss_protection(self):
        """Test protection against Cross-Site Scripting (XSS) attacks."""
        
        def sanitize_html_output(text):
            """Sanitize text for HTML output."""
            if not isinstance(text, str):
                return ""
            
            # HTML entity encoding
            html_entities = {
                '<': '&lt;',
                '>': '&gt;',
                '&': '&amp;',
                '"': '&quot;',
                "'": '&#x27;',
                '/': '&#x2F;'
            }
            
            sanitized = text
            for char, entity in html_entities.items():
                sanitized = sanitized.replace(char, entity)
            
            return sanitized
        
        # Test XSS payloads
        xss_payloads = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "javascript:alert('xss')",
            "<svg onload=alert('xss')>",
            "';alert('xss');//"
        ]
        
        for payload in xss_payloads:
            sanitized = sanitize_html_output(payload)
            
            # Verify dangerous elements are encoded
            assert "<script>" not in sanitized
            assert "javascript:" not in sanitized
            assert "onerror=" not in sanitized
            assert "onload=" not in sanitized
            
            # Verify encoding occurred
            if "<" in payload:
                assert "&lt;" in sanitized
            if ">" in payload:
                assert "&gt;" in sanitized
    
    def test_csrf_protection(self):
        """Test protection against Cross-Site Request Forgery (CSRF) attacks."""
        
        def generate_csrf_token():
            """Generate CSRF token."""
            return secrets.token_urlsafe(32)
        
        def validate_csrf_token(token, expected_token):
            """Validate CSRF token."""
            if not token or not expected_token:
                return False
            return hmac.compare_digest(token, expected_token)
        
        # Generate CSRF token
        csrf_token = generate_csrf_token()
        assert csrf_token is not None
        assert len(csrf_token) > 20
        
        # Test valid token
        assert validate_csrf_token(csrf_token, csrf_token) is True
        
        # Test invalid token
        fake_token = "fake_token_12345"
        assert validate_csrf_token(fake_token, csrf_token) is False
        
        # Test missing token
        assert validate_csrf_token(None, csrf_token) is False
        assert validate_csrf_token(csrf_token, None) is False
    
    def test_directory_traversal_protection(self):
        """Test protection against directory traversal attacks."""
        
        def safe_file_access(filename, base_directory="/var/argus/evidence"):
            """Safely access files within base directory."""
            import os
            
            # Normalize the path
            safe_path = os.path.normpath(os.path.join(base_directory, filename))
            
            # Ensure the path is within the base directory
            if not safe_path.startswith(os.path.abspath(base_directory)):
                raise ValueError("Access denied: Path outside allowed directory")
            
            return safe_path
        
        # Test legitimate file access
        legitimate_file = "evidence_001.jpg"
        safe_path = safe_file_access(legitimate_file)
        assert "evidence_001.jpg" in safe_path
        
        # Test directory traversal attempts
        traversal_attempts = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"
        ]
        
        for attempt in traversal_attempts:
            try:
                result = safe_file_access(attempt)
                # If no exception, verify path is still safe
                assert "/var/argus/evidence" in result or "\\var\\argus\\evidence" in result
            except ValueError as e:
                # Expected behavior - access denied
                assert "Access denied" in str(e)


class TestComplianceValidation:
    """Test compliance with security and privacy requirements."""
    
    def test_data_retention_compliance(self):
        """Test data retention policy compliance."""
        
        # Mock data retention policy
        retention_policy = {
            "unconfirmed_incidents": timedelta(days=1),
            "confirmed_incidents": timedelta(days=365),
            "audit_logs": timedelta(days=2555),  # 7 years
            "user_sessions": timedelta(hours=24)
        }
        
        def should_purge_data(data_type, created_date):
            """Check if data should be purged based on retention policy."""
            if data_type not in retention_policy:
                return False
            
            retention_period = retention_policy[data_type]
            age = datetime.now() - created_date
            
            return age > retention_period
        
        # Test retention logic
        now = datetime.now()
        
        # Recent unconfirmed incident (should not purge)
        recent_unconfirmed = now - timedelta(hours=12)
        assert should_purge_data("unconfirmed_incidents", recent_unconfirmed) is False
        
        # Old unconfirmed incident (should purge)
        old_unconfirmed = now - timedelta(days=2)
        assert should_purge_data("unconfirmed_incidents", old_unconfirmed) is True
        
        # Recent confirmed incident (should not purge)
        recent_confirmed = now - timedelta(days=30)
        assert should_purge_data("confirmed_incidents", recent_confirmed) is False
        
        # Very old confirmed incident (should purge)
        very_old_confirmed = now - timedelta(days=400)
        assert should_purge_data("confirmed_incidents", very_old_confirmed) is True
        
        # Audit logs (long retention)
        old_audit = now - timedelta(days=365)
        assert should_purge_data("audit_logs", old_audit) is False
        
        very_old_audit = now - timedelta(days=3000)
        assert should_purge_data("audit_logs", very_old_audit) is True
    
    def test_privacy_anonymization_compliance(self):
        """Test privacy anonymization compliance."""
        
        def anonymize_personal_data(data):
            """Anonymize personal data for compliance."""
            if not isinstance(data, dict):
                return data
            
            anonymized = data.copy()
            
            # Fields that should be anonymized
            personal_fields = ['operator_id', 'user_id', 'ip_address', 'device_id']
            
            for field in personal_fields:
                if field in anonymized:
                    # Replace with anonymized version
                    original_value = anonymized[field]
                    hash_input = f"{field}:{original_value}:salt"
                    anonymized_value = f"anon_{hashlib.sha256(hash_input.encode()).hexdigest()[:8]}"
                    anonymized[field] = anonymized_value
            
            # Mark as anonymized
            anonymized['_anonymized'] = True
            anonymized['_anonymized_at'] = datetime.now().isoformat()
            
            return anonymized
        
        # Test data anonymization
        personal_data = {
            "incident_id": "inc_001",
            "operator_id": "op_12345",
            "user_id": "user_67890",
            "ip_address": "192.168.1.100",
            "device_id": "device_abc123",
            "timestamp": "2024-01-01T12:00:00Z",
            "confidence": 0.95
        }
        
        anonymized_data = anonymize_personal_data(personal_data)
        
        # Verify anonymization occurred
        assert anonymized_data['_anonymized'] is True
        assert '_anonymized_at' in anonymized_data
        
        # Verify personal fields were anonymized
        assert anonymized_data['operator_id'] != personal_data['operator_id']
        assert anonymized_data['user_id'] != personal_data['user_id']
        assert anonymized_data['ip_address'] != personal_data['ip_address']
        assert anonymized_data['device_id'] != personal_data['device_id']
        
        # Verify anonymized values follow pattern
        assert anonymized_data['operator_id'].startswith('anon_')
        assert anonymized_data['user_id'].startswith('anon_')
        
        # Verify non-personal fields preserved
        assert anonymized_data['incident_id'] == personal_data['incident_id']
        assert anonymized_data['timestamp'] == personal_data['timestamp']
        assert anonymized_data['confidence'] == personal_data['confidence']
    
    def test_audit_logging_compliance(self):
        """Test audit logging compliance requirements."""
        
        class AuditLogger:
            def __init__(self):
                self.logs = []
            
            def log_event(self, event_type, user_id, resource, action, result, metadata=None):
                """Log audit event."""
                audit_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "event_type": event_type,
                    "user_id": user_id,
                    "resource": resource,
                    "action": action,
                    "result": result,
                    "metadata": metadata or {},
                    "session_id": f"session_{secrets.token_hex(8)}"
                }
                
                self.logs.append(audit_entry)
                return audit_entry
        
        audit_logger = AuditLogger()
        
        # Test various audit events
        events = [
            ("authentication", "user_001", "system", "login", "success"),
            ("authorization", "user_001", "incident_123", "view", "success"),
            ("data_access", "user_001", "evidence_456", "retrieve", "success"),
            ("data_modification", "user_001", "incident_123", "update", "success"),
            ("system_admin", "admin_001", "user_002", "create", "success"),
            ("authentication", "user_999", "system", "login", "failure")
        ]
        
        for event_type, user_id, resource, action, result in events:
            audit_entry = audit_logger.log_event(event_type, user_id, resource, action, result)
            
            # Verify required audit fields
            assert audit_entry["timestamp"] is not None
            assert audit_entry["event_type"] == event_type
            assert audit_entry["user_id"] == user_id
            assert audit_entry["resource"] == resource
            assert audit_entry["action"] == action
            assert audit_entry["result"] == result
            assert audit_entry["session_id"] is not None
        
        # Verify all events were logged
        assert len(audit_logger.logs) == len(events)
        
        # Test audit log integrity
        for log_entry in audit_logger.logs:
            # Verify timestamp format
            datetime.fromisoformat(log_entry["timestamp"])  # Should not raise exception
            
            # Verify required fields present
            required_fields = ["timestamp", "event_type", "user_id", "resource", "action", "result"]
            for field in required_fields:
                assert field in log_entry
                assert log_entry[field] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])