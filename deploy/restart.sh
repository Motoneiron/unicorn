#!/bin/bash
set -e
grep -q '^TELEGRAM_PROXY=' /opt/unicorn/.env || echo 'TELEGRAM_PROXY=http://127.0.0.1:8790' >> /opt/unicorn/.env
cp /opt/unicorn/unicorn.service /etc/systemd/system/unicorn.service
systemctl daemon-reload
systemctl restart unicorn
sleep 8
systemctl is-active unicorn
journalctl -u unicorn -n 30 --no-pager
