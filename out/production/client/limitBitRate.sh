#!/bin/bash
# change enps20 to the right interface (e.g. eth0)
sudo tc qdisc del dev enp2s0 root
sudo tc qdisc add dev enp2s0 root handle 1:0 htb default 10
sudo tc class add dev enp2s0 parent 1:0 classid 1:10 htb rate 120kbps prio 0
sudo iptables -A OUTPUT -t mangle -p tcp --sport 80 -j MARK --set-mark 10
sudo tc filter add dev enp2s0 parent 1:0 prio 0 protocol ip handle 10 fw flowid 1:10
