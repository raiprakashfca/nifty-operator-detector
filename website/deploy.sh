#!/bin/bash
# Deploy website/index.html to the nginx web root on the VPS and check TLS/DNS.
URL=https://raw.githubusercontent.com/raiprakashfca/nifty-operator-detector/claude/admiring-planck-nvpl8v/website/index.html
echo "== 1. find web root =="
ROOT=$(nginx -T 2>/dev/null | awk '/server_name/ && /apnaalgo\.ai/ {f=1} f && $1=="root" {gsub(";","",$2); print $2; exit}')
echo "ROOT=$ROOT"
if [ -z "$ROOT" ] || [ ! -d "$ROOT" ]; then
  nginx -T 2>/dev/null | grep -nE 'server_name|root|listen'
  echo "STOPPING - paste this output back to Claude"; exit 1
fi
echo "== 2. backup + deploy =="
BK=/root/site-backup-$(date +%Y%m%d-%H%M%S)
cp -a "$ROOT" "$BK" && echo "backup: $BK"
curl -fsSL "$URL" -o "$ROOT/index.html.new" || { echo "DOWNLOAD FAILED"; exit 1; }
mv "$ROOT/index.html.new" "$ROOT/index.html"
chown --reference="$ROOT" "$ROOT/index.html"; chmod 644 "$ROOT/index.html"
nginx -t && systemctl reload nginx
echo "== 3. local check (want 200 48508) =="
curl -s -o /dev/null -w "http %{http_code} %{size_download}\n" -H "Host: apnaalgo.ai" http://127.0.0.1/
curl -sk -o /dev/null -w "https %{http_code} %{size_download}\n" --resolve apnaalgo.ai:443:127.0.0.1 https://apnaalgo.ai/
echo "== 4. TLS cert =="
echo | openssl s_client -connect 127.0.0.1:443 -servername apnaalgo.ai 2>/dev/null | openssl x509 -noout -subject -issuer -dates -ext subjectAltName
certbot certificates 2>/dev/null | grep -E 'Certificate Name|Domains|Expiry' || echo "certbot not installed"
systemctl list-timers 2>/dev/null | grep -i certbot || echo "no certbot timer"
echo "== 5. DNS =="
echo "this server: $(curl -s -4 ifconfig.me)"
echo "apnaalgo.ai -> $(getent hosts apnaalgo.ai | awk '{print $1}')"
echo "www -> $(getent hosts www.apnaalgo.ai | awk '{print $1}')"
echo "DONE - paste everything above back to Claude"
