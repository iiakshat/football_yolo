from ultralytics import YOLO

model = YOLO("models/best.pt")

result = model.predict("matches/vid1.mp4", save=True)
print(result[0])
print("+" * 150)

for i,box in enumerate(result[0].boxes):
    print(box)
    print("+" * 70, f" Box {i}", "+" * 70, )