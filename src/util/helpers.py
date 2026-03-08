import os
import openroad as ord
from openroad import Tech, Design, Timing
import odb


def get_rss_mb():
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / 1024.0
    except FileNotFoundError:
        return None
    return None

def is_circuit_input(iterm: odb.dbITerm) -> bool:
    net = iterm.getNet()
    if net is None:
        return False


    sig = net.getSigType()
    
    if(sig == "CLOCK" or sig == "POWER" or sig == "GROUND" or iterm.getMTerm().getName().upper().find("CLK") != -1):
        return False

    for bterm in net.getBTerms():
        io = bterm.getIoType()
        bsig = bterm.getSigType()
        if (io=="INPUT" or io=="INOUT") \
           and not bsig=="POWER" \
           and not bsig=="GROUND":
            return True

    return False

def is_circuit_output(iterm: odb.dbITerm) -> bool:
    net = iterm.getNet()
    if net is None:
        return False

    sig = net.getSigType()
    
    if(sig == "CLOCK" or sig == "POWER" or sig == "GROUND"):
        return False

    # assume there's an output net, Y->A   the actual output is A; 
    # assume there's an output net, Y (1 pin net)  the actual output is Y
    if isDriverPin(iterm):
        for pin in iterm.getNet().getITerms():
            if pin == iterm:
                continue
            if not isDriverPin(pin):            
                return False

    for bterm in net.getBTerms():
        io = bterm.getIoType()
        bsig = bterm.getSigType()
        if (io=="OUTPUT" or io=="INOUT") \
           and not bsig=="POWER" \
           and not bsig=="GROUND":
            return True

    
    return False

def isDriverPin(Pin) -> bool:   #accept Pin class or dbITerm
    if Pin is None:
        return False

    iterm = getattr(Pin, "db_ITerm", Pin)  

    net = iterm.getNet()
    if net is None:
        return False

    sig = net.getSigType()
    if sig in ("CLOCK", "POWER", "GROUND"):
        return False

    return bool(iterm.isOutputSignal())

def get_time_unit(design):
    s = design.evalTclString('sta::unit_scaled_suffix time')
    unit = 0
    if s.strip() == "ps":
        unit = 1e-12
    elif s.strip() == "ns":
        unit = 1e-9
    elif s.strip() == "us":
        unit = 1e-6
    elif s.strip() == "ms":
        unit = 1e-3
    elif s.strip() == "s":
        unit = 1.0
    elif s.strip() == "fs":
        unit = 1e-15
    else:
        print(f"[ERROR] Unknown time unit: {s.strip()}")
        os._exit(1)
    return unit

def isSignalNet(net_type):
    if net_type == "CLOCK" or net_type == "POWER" or net_type == "GROUND":
        return False
    return True

def isClock(net_type):
    return net_type == "CLOCK"
