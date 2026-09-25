#!/bin/bash
# After DNS points at this server: get a Let's Encrypt cert for apnaalgo.ai and verify HTTPS.
ME=$(curl -s -4 ifconfig.me)
A=$(dig +short @8.8.8.8 apnaalgo.ai A 2>/dev/null | tail -1); [ -z "$A" ] && A=$(getent ahostsv4 apnaalgo.ai | awk 'NR==1{print $1}')
W=$(dig +short @8.8.8.8 www.apnaalgo.ai A 2>/dev/null | tail -1); [ -z "$W" ] && W=$(getent ahostsv4 www.apnaalgo.ai | awk 'NR==1{print $1}')
echo "this server: $ME   apnaalgo.ai -> $A   www -> $W"
if [ "$A" != "$ME" ]; then
  echo "DNS not switched yet - wait 10 minutes and run this again. STOPPING"; exit 1
fi
DOMS="-d apnaalgo.ai"; [ "$W" = "$ME" ] && DOMS="$DOMS -d www.apnaalgo.ai" || echo "www not switched yet - getting cert for apnaalgo.ai only"
certbot --nginx $DOMS --redirect --keep-until-expiring -n --agree-tos --register-unsafely-without-email || { echo "CERTBOT FAILED - paste this to Claude"; exit 1; }
sleep 2
echo "== check =="
curl -s -o /dev/null -w "https apnaalgo %{http_code} %{size_download}\n" https://apnaalgo.ai/
curl -s -o /dev/null -w "http redirect %{http_code} -> %{redirect_url}\n" http://apnaalgo.ai/
curl -sk -o /dev/null -w "gistdesk https %{http_code}\n" --resolve gistdesk.in:443:127.0.0.1 https://gistdesk.in/
certbot renew --dry-run 2>&1 | tail -3
echo "DONE - paste everything above back to Claude"
