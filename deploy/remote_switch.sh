#!/bin/bash
set -e
/opt/unicorn/venv/bin/python -c 'import ccxt,pandas,numpy; print("deps-ok")'
test -f /opt/unicorn/unicorn_bot.py
test -s /opt/unicorn/.env
systemctl stop cisdbotbybit
systemctl disable cisdbotbybit
cp /opt/unicorn/unicorn.service /etc/systemd/system/unicorn.service
systemctl daemon-reload
systemctl enable unicorn
systemctl restart unicorn
sleep 4
systemctl is-active unicorn
