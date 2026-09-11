import json

with open("mp/train/_annotations.coco.json", "r") as f:
    data = json.load(f)

print("Images:", len(data["images"]))
print("Annotations:", len(data["annotations"]))
print("Categories:")

for category in data["categories"]:
    print(category["id"], "->", category["name"])