from ultralytics import YOLO
from collections import Counter
from pathlib import Path
import os

dir_path = os.path.dirname(os.path.abspath(__file__))

def detect_objects(model: YOLO, image_path: str)-> None:
    print("[INFO] Запускаем распознавание объектов.")
    results = model(image_path, verbose=False)[0]
    
    if results.names and results.boxes is not None:
        labels = results.boxes.cls.tolist()
        label_names = [results.names[int(cls)] for cls in labels]
        counts = Counter(label_names)
        print("[INFO] Обнаруженные объекты: ")

        for label, count in counts.items():
            print(f"[+] {label}: {count}")
    else:
        print("[!] Объекты не обнаружены!")

    save_path = results.save(filename=f"RESULT_{Path(image_path).stem}.png")
    print(f"[INFO] Результат сохранён в файл: {save_path}")

def pick_image():
    global img_path
    img_path = input("Введите полный путь до своего изображения >>> ")

def main():
    model_select = str(input("Вас приветствует программа Cascad Eye\nВыберите модель для распознавания (1/2/3) >>> "))
    if model_select == "1":
        model = YOLO(dir_path + r"\models\yolov8n.pt")
    elif model_select == "2":
        model = YOLO(dir_path + r"\models\yolo11n.pt")
    elif model_select == "3":
        model = YOLO(dir_path + r"\models\yolo11x.pt")
    else:
        print("[ERROR] Введена некорректная информация! Пожалуйста, повторите попытку!")
   
    pick_image()
    detect_objects(model, img_path)


if __name__ == "__main__":
    main()
