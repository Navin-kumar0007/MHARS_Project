from mhars.core import MHARS
import json

pipeline = MHARS(machine_type_id=0) # CPU
res = pipeline.run(56.9)
print(json.dumps(res.metadata, indent=2))
