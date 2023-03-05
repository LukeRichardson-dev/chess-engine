from base64 import b64decode

data = "AkAEBQJABkABAUBABAEBQEBAA0BAA0ABQEBAAUBAQEBAQEDDQEBAQEBAw0DEQMFAwcFAQMHBxMFAQMLFQMLGQA=="
b = b64decode(data)

buf = ""
for idx, c in enumerate(b):
    c2 = c & 0b1111
    side = "w" if c & 0b10000000 != 0 else "b"
    if idx % 8 == 0:
        buf += "\n" \
            # + "-" * 24 + "\n"

    if c2 == 1:
        buf += "p" + side
    elif c2 == 2:
        buf += "R" + side
    elif c2 == 3:
        buf += "N" + side
    elif c2 == 4:
        buf += "B" + side
    elif c2 == 5:
        buf += "Q" + side
    elif c2 == 6:
        buf += "K" + side
    else:
        buf += "  "

    buf += "|"

print(b)
print(buf)