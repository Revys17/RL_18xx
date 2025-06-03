# Remote Access Instructions for RL18xx Training

This guide explains how to securely access your training dashboard and TensorBoard from your local machine while running training on remote hardware.

## Services and Ports

- **Dashboard**: Port 5001
- **TensorBoard**: Port 6006
- **Both services bind to 0.0.0.0** (accessible from any network interface)

## Method 1: SSH Port Forwarding (Recommended)

This is the most secure method as it tunnels traffic through SSH.

### Step 1: Connect with SSH Port Forwarding

From your local machine, connect to your remote server with port forwarding:

```bash
ssh -L 5001:localhost:5001 -L 6006:localhost:6006 user@your-server-ip
```

Replace `user` with your username and `your-server-ip` with your server's IP address.

### Step 2: Access Services Locally

Once connected, you can access:
- Dashboard: http://localhost:5001
- TensorBoard: http://localhost:6006

### Using SSH Config (Optional)

Add this to your `~/.ssh/config` file for easier access:

```
Host rl18xx-training
    HostName your-server-ip
    User your-username
    LocalForward 5001 localhost:5001
    LocalForward 6006 localhost:6006
```

Then connect with: `ssh rl18xx-training`

## Method 2: Reverse SSH Tunnel (For Restricted Networks)

If your remote server is behind a firewall, use a reverse tunnel:

From the remote server:
```bash
ssh -R 5001:localhost:5001 -R 6006:localhost:6006 user@your-home-ip
```

## Method 3: VPN Access

If you have a VPN set up between your local machine and the remote server, you can access the services directly using the server's VPN IP:
- Dashboard: http://server-vpn-ip:5001
- TensorBoard: http://server-vpn-ip:6006

## Security Considerations

1. **Never expose these ports directly to the internet** without proper authentication
2. Use SSH tunneling for the most secure access
3. If you must expose ports, use a reverse proxy (nginx) with authentication
4. Consider using a VPN for regular access

## Firewall Configuration (If Needed)

If using cloud providers, ensure your security groups/firewall rules:
- Keep ports 5001 and 6006 **closed** to the public internet
- Only allow SSH (port 22) from your IP address

## Starting the Services

1. SSH into your remote server
2. Navigate to the project directory: `cd /path/to/RL_18xx`
3. Run: `./startup.sh`
4. Services will be available through your SSH tunnel

## Troubleshooting

### Connection Refused
- Ensure services are running: `ps aux | grep -E 'tensorboard|gunicorn'`
- Check if ports are listening: `netstat -tlnp | grep -E '5001|6006'`

### SSH Tunnel Not Working
- Add `-v` flag for verbose output: `ssh -v -L 5001:localhost:5001 ...`
- Ensure `AllowTcpForwarding` is enabled in server's `/etc/ssh/sshd_config`

### Can't Access After Disconnecting SSH
- SSH tunnels only work while the SSH connection is active
- Use `screen` or `tmux` to keep the SSH session alive
- Or use `autossh` for persistent tunnels:
  ```bash
  autossh -M 0 -f -N -L 5001:localhost:5001 -L 6006:localhost:6006 user@server
  ```

## Advanced: Nginx Reverse Proxy with Authentication

For a more permanent solution with basic authentication:

1. Install nginx: `sudo apt install nginx apache2-utils`
2. Create password file: `sudo htpasswd -c /etc/nginx/.htpasswd your-username`
3. Configure nginx (`/etc/nginx/sites-available/rl18xx`):

```nginx
server {
    listen 80;
    server_name your-domain.com;

    auth_basic "RL18xx Training";
    auth_basic_user_file /etc/nginx/.htpasswd;

    location /dashboard/ {
        proxy_pass http://localhost:5001/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /tensorboard/ {
        proxy_pass http://localhost:6006/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

4. Enable site: `sudo ln -s /etc/nginx/sites-available/rl18xx /etc/nginx/sites-enabled/`
5. Reload nginx: `sudo nginx -s reload`