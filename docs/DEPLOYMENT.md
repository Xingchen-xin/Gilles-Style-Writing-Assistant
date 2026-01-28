# GSWA Deployment Guide

This guide covers how to deploy GSWA for external access, allowing your team to use the web interface from anywhere.

## Authentication (Recommended for Public Access)

Enable password protection by setting environment variables:

```bash
# Set username and password
export GSWA_AUTH_USER=gilles
export GSWA_AUTH_PASS=your-secret-password

# Then start the server
make run-external
```

Or add to `.env` file:
```env
GSWA_AUTH_USER=gilles
GSWA_AUTH_PASS=your-secret-password
```

When authentication is enabled:
- Browser will prompt for username/password on first visit
- Credentials are remembered for the session
- Health check endpoint (`/v1/health`) is always accessible without auth

## Quick Start

### Option 1: Local Network Access (Recommended for Lab/Office)

```bash
# Start the server on all interfaces
make run-external

# Or manually:
python -m gswa.api --host 0.0.0.0 --port 8000
```

Access the app at: `http://<server-ip>:8000`

To find your server's IP:
```bash
hostname -I | awk '{print $1}'
```

### Option 2: SSH Tunnel (Secure Remote Access)

On your local machine:
```bash
# Replace <server> with your server's hostname/IP
ssh -L 8000:localhost:8000 user@<server>

# Then access locally at: http://localhost:8000
```

### Option 3: Cloudflare Tunnel (Public HTTPS - Recommended)

Cloudflare Tunnel provides secure public HTTPS access without exposing ports.

```bash
# Install cloudflared
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb

# Quick tunnel (temporary URL, no account needed)
cloudflared tunnel --url http://localhost:8000

# This gives you a URL like: https://xxxx-xxxx-xxxx.trycloudflare.com
```

For persistent tunnels:
```bash
# Login to Cloudflare (one-time)
cloudflared tunnel login

# Create a named tunnel
cloudflared tunnel create gswa

# Configure the tunnel
cat > ~/.cloudflared/config.yml << EOF
tunnel: gswa
credentials-file: /home/$USER/.cloudflared/<tunnel-id>.json
ingress:
  - hostname: gswa.yourdomain.com
    service: http://localhost:8000
  - service: http_status:404
EOF

# Run the tunnel
cloudflared tunnel run gswa
```

### Option 4: ngrok (Alternative Public Access)

```bash
# Install ngrok
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | \
  sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | \
  sudo tee /etc/apt/sources.list.d/ngrok.list
sudo apt update && sudo apt install ngrok

# Authenticate (free account required)
ngrok config add-authtoken <your-token>

# Start tunnel
ngrok http 8000
```

## Production Deployment

### Using systemd (Auto-start on Boot)

Create a systemd service file:

```bash
sudo tee /etc/systemd/system/gswa.service << EOF
[Unit]
Description=GSWA - Gilles Style Writing Assistant
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment="PATH=/home/$USER/micromamba/envs/gswa/bin:\$PATH"
ExecStart=/home/$USER/micromamba/envs/gswa/bin/python -m gswa.api --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable gswa
sudo systemctl start gswa

# Check status
sudo systemctl status gswa
```

### Using Docker (Alternative)

```bash
# Build the Docker image
docker build -t gswa .

# Run with GPU support
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  --name gswa \
  gswa
```

### Nginx Reverse Proxy (with HTTPS)

```nginx
server {
    listen 80;
    server_name gswa.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    server_name gswa.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/gswa.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/gswa.yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 300s;
    }
}
```

## Starting Services

### Complete Startup Script

```bash
# Start vLLM server (required for generation)
make start-vllm

# Start GSWA API server
make run-external

# Or run both in tmux
make run-production
```

### Makefile Commands

| Command | Description |
|---------|-------------|
| `make run` | Start API on localhost:8000 |
| `make run-external` | Start API on 0.0.0.0:8000 (network accessible) |
| `make run-production` | Start both vLLM and API in tmux |
| `make start-vllm` | Start vLLM inference server |
| `make stop-vllm` | Stop vLLM server |

## Environment Variables

Configure in `.env`:

```env
# API Configuration
HOST=0.0.0.0
PORT=8000

# Model Configuration
BASE_MODEL=mistralai/Mistral-Nemo-Instruct-2407
LORA_ADAPTER_PATH=./models/gswa-lora-Mistral-20260126-131024

# vLLM Configuration
VLLM_HOST=localhost
VLLM_PORT=8001
```

## Security Considerations

1. **Authentication**: For public deployments, consider adding authentication:
   - Use Cloudflare Access for zero-trust authentication
   - Add HTTP Basic Auth via nginx
   - Implement API key authentication

2. **Rate Limiting**: Add rate limiting in nginx:
   ```nginx
   limit_req_zone $binary_remote_addr zone=gswa:10m rate=10r/m;

   location / {
       limit_req zone=gswa burst=5;
       proxy_pass http://127.0.0.1:8000;
   }
   ```

3. **Firewall**: Only open necessary ports:
   ```bash
   sudo ufw allow 22/tcp    # SSH
   sudo ufw allow 80/tcp    # HTTP (redirect)
   sudo ufw allow 443/tcp   # HTTPS
   sudo ufw enable
   ```

## Monitoring

### View Logs

```bash
# API logs
journalctl -u gswa -f

# Or if running with make
tail -f logs/api.log
```

### Health Check

```bash
curl http://localhost:8000/v1/health
```

## Troubleshooting

### Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000

# Kill if necessary
kill -9 <PID>
```

### Connection Refused
- Verify the server is running: `systemctl status gswa`
- Check firewall: `sudo ufw status`
- Verify binding address: use `0.0.0.0` for external access

### Slow Generation
- Check GPU memory: `nvidia-smi`
- Verify vLLM is running: `curl http://localhost:8001/health`
- Consider reducing `n_variants` or `max_tokens`

## Feedback Collection

The web UI automatically collects user feedback for model improvement:

1. Users rate variants as **Best**, **Good**, **Bad**, or **Edit** them
2. Feedback is stored in `logs/feedback/`
3. Export for DPO training via the **Feedback Stats** tab or API:
   ```bash
   curl -X POST http://localhost:8000/v1/feedback/export-dpo
   ```
4. Re-train with collected preferences:
   ```bash
   python scripts/train_dpo.py data/training/dpo_pairs.jsonl
   ```
