#!/bin/bash
# Read-only: show who serves ports 80/443 and how nginx is wired. Changes nothing.
echo "== listeners on 80/443 =="
ss -ltnp 2>/dev/null | grep -E ':(80|443)\s'
echo "== nginx includes =="
grep -n 'include' /etc/nginx/nginx.conf
echo "== sites-enabled / conf.d =="
ls -l /etc/nginx/sites-enabled /etc/nginx/conf.d 2>&1
echo "== loaded server blocks (file, listen, server_name, root, proxy_pass) =="
nginx -T 2>/dev/null | grep -nE '^# configuration file|^\s*(listen|server_name|root|proxy_pass)\b'
echo "== docker =="
command -v docker >/dev/null && docker ps --format '{{.Names}}  {{.Image}}  {{.Ports}}' || echo "no docker"
echo "== other web servers =="
systemctl is-active apache2 caddy traefik openresty 2>/dev/null
echo "== 404 body =="
curl -s -H "Host: apnaalgo.ai" http://127.0.0.1/ | head -c 300; echo
curl -sI -H "Host: apnaalgo.ai" http://127.0.0.1/ | grep -iE '^(HTTP|server)'
echo "DONE - paste everything above back to Claude"
