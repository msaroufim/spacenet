import os
import sys

DEST_DIR     = sys.argv[2]
SOURCE_FILE   = sys.argv[1]
NUM_EXAMPLES = sys.argv[3]

with open(SOURCE_FILE,'r') as f:
    txt = f.read()

    lines = txt.splitlines()

    i = 0
    for line in lines:
        os.system("curl " + line + "> " + DEST_DIR + "" "img" + str(i) +
                 ".tif")
        i += 1
        if i > NUM_EXAMPLES:
            break
