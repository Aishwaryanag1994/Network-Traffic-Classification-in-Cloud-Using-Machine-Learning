
from typing import Optional

HTTP_PORTS  = {80, 8080, 8000, 8888}
HTTPS_PORTS = {443, 8443, 9443, 993, 995}
DNS_PORTS   = {53}
VOIP_SIGNAL = {5060, 5061}              
VOIP_RTP_UDP_RANGE = range(16384, 32768)  
P2P_PORTS   = {6881, 6882, 6883, 51413} 

def protocol_class(proto_num: int, sport: Optional[int], dport: Optional[int]) -> str:
    sport = int(sport) if sport not in (None, "") else -1
    dport = int(dport) if dport not in (None, "") else -1

    if dport in HTTPS_PORTS or sport in HTTPS_PORTS:
        return "ENCRYPTED"
    if dport in HTTP_PORTS or sport in HTTP_PORTS:
        return "HTTP"
    if dport in DNS_PORTS or sport in DNS_PORTS:
        return "DNS"
    if dport in VOIP_SIGNAL or sport in VOIP_SIGNAL:
        return "VOIP"
    if proto_num == 17 and (dport in VOIP_RTP_UDP_RANGE or sport in VOIP_RTP_UDP_RANGE):
        return "VOIP"
    if dport in P2P_PORTS or sport in P2P_PORTS:
        return "P2P"
    return "OTHER"
