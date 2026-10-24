#!/bin/bash
# Hold the NVIDIA driver and the kernel during a long run (docs/longrun_runbook.md, "Host stability").
#
#   host_hold.sh status          what is held, the loaded vs installed NVIDIA versions, pending upgrades
#                                and reboots (no sudo)
#   host_hold.sh check           exit 0 if every package hold would cover is held, else 1 and print the ones
#                                that aren't (no sudo; the daily report's M7)
#   sudo host_hold.sh hold       apt-mark hold the installed NVIDIA driver-series packages and the kernel
#                                metapackages (unattended-upgrades respects holds)
#   sudo host_hold.sh unhold     undo, after the run (then upgrade and reboot in one go)
#
# Why: unattended-upgrades upgraded this machine's NVIDIA driver on 2026-05-22, 07-24 and 10-10. The new
# userspace libraries don't match the kernel module still loaded, so every process started afterwards gets
# "Failed to initialize NVML: Driver/library version mismatch" -- a run that restarts after a crash loses the
# GPU until a reboot. Holding the kernel keeps a reboot (planned or a power cut) on the kernel the driver's
# DKMS module was built for.
set -euo pipefail

# The driver series' packages (the "-595" ones and the firmware) and the kernel metapackages, as installed.
packages() {
    dpkg-query -W -f '${db:Status-Abbrev} ${Package}\n' 2>/dev/null | awk '$1 == "ii" {print $2}' | sed 's/:amd64$//' \
        | grep -E '^((lib)?nvidia-.*-[0-9]{3}(-open)?|nvidia-firmware-[0-9]{3}-.*|xserver-xorg-video-nvidia-[0-9]{3}|linux-(generic|image-generic|headers-generic|modules-nvidia-.*-generic)(-hwe-[0-9.]+)?)$' \
        | sort -u
}

case ${1:-status} in
status)
    echo "kernel running   $(uname -r)"
    if [ -r /proc/driver/nvidia/version ]; then
        echo "nvidia module    $(sed -n '1s/.*Kernel Module *\(for [^ ]* *\)\{0,1\}\([0-9.]*\).*/\2/p' /proc/driver/nvidia/version)"
    fi
    echo "nvidia libraries $(dpkg-query -W -f '${Version}\n' 'libnvidia-compute-*' 2>/dev/null | sort -u | tr '\n' ' ')"
    held=$(apt-mark showhold)
    echo "held packages    ${held:-none}" | tr '\n' ' '; echo
    missing=$(comm -23 <(packages) <(echo "$held" | sort -u) | tr '\n' ' ')
    echo "not held         ${missing:-none (all held)}"
    upgradable=$(apt list --upgradable 2>/dev/null | grep -E '^((lib)?nvidia|linux-)' | cut -d/ -f1 | tr '\n' ' ')
    echo "upgradable now   ${upgradable:-none}"
    echo "auto upgrades    $(apt-config dump APT::Periodic::Unattended-Upgrade | sed 's/.*"\(.*\)";/\1/') (1 = on)"
    [ -e /var/run/reboot-required ] && echo "REBOOT REQUIRED  $(cat /var/run/reboot-required.pkgs 2>/dev/null | tr '\n' ' ')"
    true
    ;;
check)
    held=$(apt-mark showhold) || exit 2
    missing=$(comm -23 <(packages) <(echo "$held" | sort -u))
    [ -z "$(packages)" ] && { echo "no driver or kernel packages found" >&2; exit 2; }
    [ -z "$missing" ] && exit 0
    echo "$missing"
    exit 1
    ;;
hold|unhold)
    [ "$(id -u)" -eq 0 ] || { echo "run with sudo" >&2; exit 1; }
    mapfile -t pkgs < <(packages)
    apt-mark "$1" "${pkgs[@]}"
    ;;
*) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
