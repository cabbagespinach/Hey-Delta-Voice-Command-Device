#!/usr/bin/env python3
"""
Send one prepared SMS (a bMessage file) through the phone over Bluetooth MAP (BlueZ obexd), in ONE process.

    python3 map_send.py AA:BB:CC:DD:EE:FF message.bmsg

Run by devices.Phone.text(); based on the owner's working send_sms.py (2026-10-02). It must be a single process:
obexd closes a MAP session as soon as the program that opened it leaves the bus, so separate `gdbus call`s
(one per step) lose the session before the message is pushed. Needs dbus-python: the Pi's own Python has it
(`sudo apt install python3-dbus`); devices.py picks that Python when the assistant's environment lacks it.

Exit 0 = the phone accepted the message for sending; otherwise the reason is printed on stderr.
"""
import sys
import time

import dbus


def main(mac: str, bmsg_path: str, timeout: float = 30.0):
    bus = dbus.SessionBus()
    client = dbus.Interface(bus.get_object("org.bluez.obex", "/org/bluez/obex"), "org.bluez.obex.Client1")
    try:
        session = client.CreateSession(mac, dbus.Dictionary({"Target": "map"}, signature="sv"))
    except dbus.DBusException as e:
        sys.exit(f"could not open the message connection (allow 'Text messages' / 'Message access' for the Pi "
                 f"in the phone's Bluetooth settings): {e.get_dbus_message() or e}")
    try:
        mas = dbus.Interface(bus.get_object("org.bluez.obex", session), "org.bluez.obex.MessageAccess1")
        mas.SetFolder("telecom/msg/outbox")
        transfer, _ = mas.PushMessage(bmsg_path, "", dbus.Dictionary(
            {"Transparent": dbus.Boolean(False), "Retry": dbus.Boolean(True), "Charset": "utf8"}, signature="sv"))
        props = dbus.Interface(bus.get_object("org.bluez.obex", transfer), "org.freedesktop.DBus.Properties")
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                status = str(props.Get("org.bluez.obex.Transfer1", "Status"))
            except dbus.DBusException:          # obexd removes a transfer once it has finished
                return
            if status == "complete":
                return
            if status == "error":
                sys.exit("the phone refused the message (transfer status: error)")
            time.sleep(0.5)
        sys.exit("sending timed out")
    except dbus.DBusException as e:
        sys.exit(f"sending failed: {e.get_dbus_message() or e}")
    finally:
        try:
            client.RemoveSession(session)
        except dbus.DBusException:
            pass


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
