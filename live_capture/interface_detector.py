import psutil
import socket
import logging

def get_available_interfaces():
    """
    Auto-detects available network interfaces using psutil.
    Returns a dictionary of interface names and their IP addresses.
    """
    interfaces = {}
    try:
        addrs = psutil.net_if_addrs()
        stats = psutil.net_if_stats()
        
        for iface_name, iface_addrs in addrs.items():
            # Check if interface is up and running
            if iface_name in stats and stats[iface_name].isup:
                ip_addr = None
                for addr in iface_addrs:
                    if addr.family == socket.AF_INET: # IPv4
                        ip_addr = addr.address
                        break
                
                interfaces[iface_name] = {
                    "ip": ip_addr or "Unknown",
                    "status": "UP"
                }
    except Exception as e:
        logging.error(f"Failed to detect network interfaces: {e}")
        
    return interfaces
