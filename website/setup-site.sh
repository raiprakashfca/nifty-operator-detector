#!/bin/bash
# Give apnaalgo.ai its own nginx site on a VPS that already hosts other sites.
# Also undoes deploy.sh's write into /var/www/html (the default site).
URL=https://raw.githubusercontent.com/raiprakashfca/nifty-operator-detector/claude/admiring-planck-nvpl8v/website/index.html
SITE=/var/www/apnaalgo.ai
CONF=/etc/nginx/sites-available/apnaalgo.ai

echo "== 1. undo earlier write to /var/www/html =="
BK=$(ls -d /root/site-backup-* 2>/dev/null | sort | head -1)
if [ -n "$BK" ] && [ -f "$BK/index.html" ]; then
  cp -a "$BK/index.html" /var/www/html/index.html && echo "restored /var/www/html/index.html from $BK"
elif [ -n "$BK" ]; then
  rm -f /var/www/html/index.html && echo "removed /var/www/html/index.html (was not there before)"
else
  echo "no backup found - leaving /var/www/html alone"
fi

echo "== 2. site files =="
mkdir -p "$SITE"
curl -fsSL "$URL" -o "$SITE/index.html" || { echo "DOWNLOAD FAILED - STOPPING"; exit 1; }
chown -R www-data:www-data "$SITE"; chmod 644 "$SITE/index.html"
ls -l "$SITE"

echo "== 3. nginx site =="
if grep -rqs 'apnaalgo\.ai' /etc/nginx/sites-enabled /etc/nginx/conf.d; then
  echo "an apnaalgo.ai config already exists - not writing a new one:"
  grep -rn 'apnaalgo\.ai' /etc/nginx/sites-enabled /etc/nginx/conf.d
else
  cat > "$CONF" <<'NGX'
server {
    listen 80;
    listen [::]:80;
    server_name apnaalgo.ai www.apnaalgo.ai;
    root /var/www/apnaalgo.ai;
    index index.html;
    location / {
        try_files $uri $uri/ /index.html;
    }
}
NGX
  ln -sf "$CONF" /etc/nginx/sites-enabled/apnaalgo.ai
fi
if nginx -t; then
  systemctl reload nginx && echo "nginx reloaded"
else
  echo "nginx config test FAILED - removing new site so nothing breaks"
  rm -f /etc/nginx/sites-enabled/apnaalgo.ai
  nginx -t && systemctl reload nginx
  echo "STOPPING - paste this output back to Claude"; exit 1
fi

echo "== 4. check (want 200 46015, gistdesk still 200) =="
curl -s -o /dev/null -w "apnaalgo http %{http_code} %{size_download}\n" -H "Host: apnaalgo.ai" http://127.0.0.1/
curl -s -o /dev/null -w "www      http %{http_code} %{size_download}\n" -H "Host: www.apnaalgo.ai" http://127.0.0.1/
curl -sk -o /dev/null -w "gistdesk https %{http_code} %{size_download}\n" --resolve gistdesk.in:443:127.0.0.1 https://gistdesk.in/
echo "DONE - paste everything above back to Claude"
