from ultralytics import YOLO
import cv2
import os

model = YOLO("model/best.pt")

helmets = [cv2.imread(os.path.join("data/helmet", fp)) for fp in os.listdir("data/helmet") if fp.startswith('helmet')]
non_helmets = [cv2.imread(os.path.join("data/helmet", fp)) for fp in os.listdir("data/helmet") if fp.startswith('non_helmet')]

helmets_results = model.predict(source=helmets)
non_helmets_results = model.predict(source=non_helmets)

model.predict(source=helmets + non_helmets, save=True)


# # accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# results = model.predict(source="0")
# results = model.predict(source="folder", show=True) # Display preds. Accepts all YOLO predict arguments
#
# # from PIL
# im1 = Image.open("bus.jpg")
# results = model.predict(source=im1, save=True)  # save plotted images
#
# # from ndarray
# im2 = cv2.imread("bus.jpg")
# results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels
#
# # from list of PIL/ndarray
# results = model.predict(source=[im1, im2])
