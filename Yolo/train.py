from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    model.train(
        data="datasets/data.yaml", epochs=50, device=[0], imgsz=640, batch=16, amp=False
    )
