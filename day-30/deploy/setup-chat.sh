#!/usr/bin/env bash
# Run from uploaded day-30 directory, after setup-server.sh. Fresh dedicated VM only.
set -euo pipefail
task_ip=${1:?Usage: bash deploy/setup-chat.sh PUBLIC_IPV4}
[[ "$task_ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo 'Expected IPv4'; exit 1; }
task_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$task_root"
echo 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4  tokenizer/tokenizer.json' | sha256sum -c -
sudo apt-get update -qq
sudo apt-get install -y nginx
sudo env UV_TOOL_DIR=/opt/certbot-tools UV_TOOL_BIN_DIR=/usr/local/bin uv tool install --python /usr/bin/python3 certbot==5.8.0
id day30 >/dev/null 2>&1 || sudo useradd --system --home-dir /var/lib/day30 --shell /usr/sbin/nologin day30
sudo install -d -m 755 /opt/ai-advent-day30 /etc/day30
sudo install -d -o day30 -g day30 -m 700 /var/lib/day30
sudo cp chat.py model.py manage.py pyproject.toml uv.lock /opt/ai-advent-day30/
sudo cp -r static tokenizer /opt/ai-advent-day30/
sudo chown -R root:root /opt/ai-advent-day30
sudo chmod -R go-w /opt/ai-advent-day30
sudo env UV_CACHE_DIR=/var/cache/day30-uv uv sync --directory /opt/ai-advent-day30 --locked --python /usr/bin/python3 --no-dev
printf 'PUBLIC_ORIGIN=https://%s\nDATABASE_PATH=/var/lib/day30/chat.sqlite3\n' "$task_ip" | sudo tee /etc/day30/chat.env >/dev/null
sudo chmod 600 /etc/day30/chat.env
sudo install -m 644 deploy/chat.service /etc/systemd/system/day30-chat.service
sudo install -m 644 deploy/certbot-renew.service deploy/certbot-renew.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now day30-chat.service
sudo systemctl restart day30-chat.service
sudo install -d -m 755 /var/www/acme
sudo rm -f /etc/nginx/sites-enabled/default
if ! sudo test -f /etc/letsencrypt/live/day30/fullchain.pem; then
    sed "s/__IP__/$task_ip/g" deploy/nginx-http.conf | sudo tee /etc/nginx/sites-available/day30 >/dev/null
    sudo ln -sf /etc/nginx/sites-available/day30 /etc/nginx/sites-enabled/day30
    sudo nginx -t
    sudo systemctl reload nginx
    sudo ufw allow 80/tcp
    sudo certbot certonly --non-interactive --agree-tos --register-unsafely-without-email \
        --webroot --webroot-path /var/www/acme --preferred-profile shortlived \
        --ip-address "$task_ip" --cert-name day30
fi
sed "s/__IP__/$task_ip/g" deploy/nginx.conf | sudo tee /etc/nginx/sites-available/day30 >/dev/null
sudo ln -sf /etc/nginx/sites-available/day30 /etc/nginx/sites-enabled/day30
sudo nginx -t
sudo systemctl enable --now nginx
sudo systemctl reload nginx
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo systemctl enable --now certbot-renew.timer
curl --fail --silent --show-error "https://$task_ip/health"
echo
echo 'Ready. Create accounts with manage.py as user day30; do not send passwords in arguments.'
