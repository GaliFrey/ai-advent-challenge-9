#!/usr/bin/env bash
# Run on a fresh Ubuntu 24.04 x86_64 VM with passwordless sudo.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[[ $(uname -m) == x86_64 ]] || { echo 'Requires x86_64.' >&2; exit 1; }
[[ $(id -u) != 0 ]] || { echo 'Run as the SSH user with sudo.' >&2; exit 1; }
sudo -n true
for tool in curl tar zstd sha256sum ufw; do
    command -v "$tool" >/dev/null || { echo "Missing: $tool" >&2; exit 1; }
done
# Preserve SSH access before enabling the firewall.
[[ ${SSH_CONNECTION##* } == 22 ]] || { echo 'Expected SSH port 22.' >&2; exit 1; }

ollama_version=0.40.2
uv_version=0.12.19
work_dir=$(mktemp -d)
trap 'rm -rf -- "$work_dir"' EXIT

echo "Downloading Ollama $ollama_version (official Linux archive)."
curl --fail --silent --show-error --location --retry 2 \
    "https://github.com/ollama/ollama/releases/download/v${ollama_version}/ollama-linux-amd64.tar.zst" \
    -o "$work_dir/ollama.tar.zst"
echo '726bee78706c281b0eeef00746efe51a044d71c592c3f0b195820707f31fdf04  ollama.tar.zst' \
    | (cd "$work_dir" && sha256sum --check)
sudo tar --zstd -xf "$work_dir/ollama.tar.zst" -C /usr/local
rm -- "$work_dir/ollama.tar.zst"

echo "Installing uv $uv_version."
curl --fail --silent --show-error --location --retry 2 \
    "https://github.com/astral-sh/uv/releases/download/${uv_version}/uv-x86_64-unknown-linux-gnu.tar.gz" \
    -o "$work_dir/uv.tar.gz"
echo '23bf5552d220e0842b65c862097b2ebaeba0064b74eda5e565e77fd25969d8c8  uv.tar.gz' \
    | (cd "$work_dir" && sha256sum --check)
tar -xzf "$work_dir/uv.tar.gz" -C "$work_dir"
sudo install -m 0755 "$work_dir/uv-x86_64-unknown-linux-gnu/uv" /usr/local/bin/uv
sudo install -m 0755 "$work_dir/uv-x86_64-unknown-linux-gnu/uvx" /usr/local/bin/uvx

if ! id ollama >/dev/null 2>&1; then
    sudo useradd --system --user-group --home-dir /var/lib/ollama --shell /usr/sbin/nologin ollama
fi
sudo install -d -o ollama -g ollama -m 0750 /var/lib/ollama/models
sudo install -m 0644 "$script_dir/ollama.service" /etc/systemd/system/ollama.service
sudo systemctl daemon-reload
sudo systemctl enable --now ollama.service
for attempt in {1..30}; do
    if curl --fail --silent http://127.0.0.1:11434/api/version; then
        break
    fi
    sleep 1
done
curl --fail --silent http://127.0.0.1:11434/api/version

sudo ufw allow 22/tcp
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw --force enable

echo 'Downloading qwen3:1.7b. This does not generate an answer.'
OLLAMA_HOST=127.0.0.1:11434 /usr/local/bin/ollama pull qwen3:1.7b
sudo systemctl is-active ollama.service
sudo systemctl is-enabled ollama.service
/usr/local/bin/uv --version
df -h /
