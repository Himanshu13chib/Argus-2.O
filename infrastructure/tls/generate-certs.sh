#!/bin/bash

# TLS Certificate Generation Script for Project Argus
# Generates self-signed certificates for development and testing

set -e

CERT_DIR="./certs"
DAYS=365
KEY_SIZE=2048

# Create certificate directory
mkdir -p $CERT_DIR

echo "Generating TLS certificates for Project Argus..."

# Generate CA private key
openssl genrsa -out $CERT_DIR/ca-key.pem $KEY_SIZE

# Generate CA certificate
openssl req -new -x509 -days $DAYS -key $CERT_DIR/ca-key.pem -out $CERT_DIR/ca-cert.pem \
    -subj "/C=IN/ST=Delhi/L=New Delhi/O=Project Argus/OU=Security/CN=Project Argus CA"

# Generate server private key
openssl genrsa -out $CERT_DIR/server-key.pem $KEY_SIZE

# Generate server certificate signing request
openssl req -new -key $CERT_DIR/server-key.pem -out $CERT_DIR/server.csr \
    -subj "/C=IN/ST=Delhi/L=New Delhi/O=Project Argus/OU=Services/CN=argus.local"

# Create extensions file for server certificate
cat > $CERT_DIR/server-extensions.conf << EOF
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req

[req_distinguished_name]

[v3_req]
basicConstraints = CA:FALSE
keyUsage = nonRepudiation, digitalSignature, keyEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = argus.local
DNS.2 = *.argus.local
DNS.3 = localhost
DNS.4 = api-gateway
DNS.5 = auth-service
DNS.6 = security-service
DNS.7 = alert-service
DNS.8 = evidence-service
DNS.9 = tracking-service
IP.1 = 127.0.0.1
IP.2 = 192.168.1.100
EOF

# Generate server certificate
openssl x509 -req -days $DAYS -in $CERT_DIR/server.csr -CA $CERT_DIR/ca-cert.pem \
    -CAkey $CERT_DIR/ca-key.pem -CAcreateserial -out $CERT_DIR/server-cert.pem \
    -extensions v3_req -extfile $CERT_DIR/server-extensions.conf

# Generate client private key
openssl genrsa -out $CERT_DIR/client-key.pem $KEY_SIZE

# Generate client certificate signing request
openssl req -new -key $CERT_DIR/client-key.pem -out $CERT_DIR/client.csr \
    -subj "/C=IN/ST=Delhi/L=New Delhi/O=Project Argus/OU=Clients/CN=argus-client"

# Generate client certificate
openssl x509 -req -days $DAYS -in $CERT_DIR/client.csr -CA $CERT_DIR/ca-cert.pem \
    -CAkey $CERT_DIR/ca-key.pem -CAcreateserial -out $CERT_DIR/client-cert.pem

# Generate DH parameters for perfect forward secrecy
openssl dhparam -out $CERT_DIR/dhparam.pem 2048

# Set appropriate permissions
chmod 600 $CERT_DIR/*-key.pem
chmod 644 $CERT_DIR/*-cert.pem $CERT_DIR/ca-cert.pem $CERT_DIR/dhparam.pem

# Clean up temporary files
rm -f $CERT_DIR/*.csr $CERT_DIR/*.srl $CERT_DIR/server-extensions.conf

echo "TLS certificates generated successfully in $CERT_DIR/"
echo ""
echo "Files created:"
echo "  - ca-cert.pem (Certificate Authority)"
echo "  - ca-key.pem (CA Private Key)"
echo "  - server-cert.pem (Server Certificate)"
echo "  - server-key.pem (Server Private Key)"
echo "  - client-cert.pem (Client Certificate)"
echo "  - client-key.pem (Client Private Key)"
echo "  - dhparam.pem (Diffie-Hellman Parameters)"
echo ""
echo "For production use, replace these with certificates from a trusted CA."