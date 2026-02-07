import cv2
import os
from typing import List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm


@dataclass
class Config:
    """Cấu hình cho face detection"""
    img_dir: str = "face_detection_dataset/images"
    lbl_dir: str = "face_detection_dataset/labels"
    log_dir: str = "face_detection_dataset/logs"
    vis_dir: str = "face_detection_dataset/visualizations"
    
    # Tham số Haar Cascade
    scale_factor: float = 1.1
    min_neighbors: int = 5
    min_size: Tuple[int, int] = (30, 30)
    
    # Tùy chọn output
    label_format: str = "yolo"  # "yolo" hoặc "pascal_voc"
    save_visualization: bool = False
    create_empty_labels: bool = True
    
    # Extensions hỗ trợ
    image_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")


class FaceDetectionLabeler:
    """Class để tạo nhãn cho dataset face detection sử dụng Haar Cascade"""
    
    def __init__(self, config: Config):
        self.config = config
        self._setup_directories()
        self._load_cascade()
        self.stats = {
            "total": 0,
            "with_faces": 0,
            "without_faces": 0,
            "errors": 0,
            "total_faces": 0
        }
        self.no_face_images = []
        self.error_images = []
        
    def _setup_directories(self):
        """Tạo các thư mục cần thiết"""
        for dir_path in [self.config.lbl_dir, self.config.log_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        if self.config.save_visualization:
            os.makedirs(self.config.vis_dir, exist_ok=True)
    
    def _load_cascade(self):
        """Load Haar Cascade classifier"""
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise RuntimeError("Không thể load Haar Cascade classifier")
    
    def _get_image_files(self) -> List[str]:
        """Lấy danh sách các file ảnh hợp lệ"""
        try:
            all_files = sorted(os.listdir(self.config.img_dir))
            return [
                f for f in all_files 
                if f.lower().endswith(self.config.image_extensions)
            ]
        except FileNotFoundError:
            raise FileNotFoundError(f"Thư mục {self.config.img_dir} không tồn tại")
    
    def _detect_faces(self, image: cv2.Mat) -> List[Tuple[int, int, int, int]]:
        """Phát hiện khuôn mặt trong ảnh"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Cải thiện chất lượng ảnh trước khi detect
        gray = cv2.equalizeHist(gray)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.config.scale_factor,
            minNeighbors=self.config.min_neighbors,
            minSize=self.config.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces
    
    def _convert_to_yolo(
        self, 
        bbox: Tuple[int, int, int, int], 
        img_width: int, 
        img_height: int
    ) -> Tuple[float, float, float, float]:
        """Chuyển đổi bbox từ Pascal VOC sang YOLO format"""
        x, y, w, h = bbox
        
        # YOLO format: <class> <x_center> <y_center> <width> <height> (normalized)
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        width = w / img_width
        height = h / img_height
        
        return x_center, y_center, width, height
    
    def _save_label(
        self, 
        label_path: str, 
        faces: List[Tuple[int, int, int, int]], 
        img_shape: Tuple[int, int]
    ):
        """Lưu nhãn vào file"""
        with open(label_path, "w", encoding="utf-8") as f:
            if len(faces) == 0:
                return  # File rỗng
            
            img_height, img_width = img_shape[:2]
            
            for bbox in faces:
                if self.config.label_format == "yolo":
                    x_c, y_c, w, h = self._convert_to_yolo(bbox, img_width, img_height)
                    # Class 0 cho face
                    f.write(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                else:  # pascal_voc
                    x, y, w, h = bbox
                    f.write(f"{x} {y} {w} {h}\n")
    
    def _visualize_detections(
        self, 
        image: cv2.Mat, 
        faces: List[Tuple[int, int, int, int]], 
        output_path: str
    ):
        """Vẽ bounding box lên ảnh và lưu"""
        vis_image = image.copy()
        
        for (x, y, w, h) in faces:
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                vis_image, 
                "Face", 
                (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                2
            )
        
        cv2.imwrite(output_path, vis_image)
    
    def _process_image(self, img_name: str) -> bool:
        """Xử lý một ảnh"""
        img_path = os.path.join(self.config.img_dir, img_name)
        
        try:
            # Đọc ảnh
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Không thể đọc ảnh: {img_name}")
            
            # Phát hiện khuôn mặt
            faces = self._detect_faces(image)
            
            # Tạo đường dẫn label
            base_name = Path(img_name).stem
            label_path = os.path.join(self.config.lbl_dir, f"{base_name}.txt")
            
            # Lưu nhãn
            if len(faces) == 0:
                self.no_face_images.append(img_name)
                if self.config.create_empty_labels:
                    self._save_label(label_path, [], image.shape)
            else:
                self._save_label(label_path, faces, image.shape)
                self.stats["with_faces"] += 1
                self.stats["total_faces"] += len(faces)
                
                # Visualize nếu cần
                if self.config.save_visualization:
                    vis_path = os.path.join(self.config.vis_dir, img_name)
                    self._visualize_detections(image, faces, vis_path)
            
            return True
            
        except Exception as e:
            self.error_images.append((img_name, str(e)))
            self.stats["errors"] += 1
            return False
    
    def process_all(self):
        """Xử lý toàn bộ dataset"""
        print("🚀 Bắt đầu tạo nhãn cho dataset...")
        print(f"📂 Thư mục ảnh: {self.config.img_dir}")
        print(f"📝 Format nhãn: {self.config.label_format.upper()}")
        print(f"⚙️  Tham số: scaleFactor={self.config.scale_factor}, "
              f"minNeighbors={self.config.min_neighbors}")
        print("-" * 60)
        
        # Lấy danh sách ảnh
        image_files = self._get_image_files()
        self.stats["total"] = len(image_files)
        
        if self.stats["total"] == 0:
            print("⚠️  Không tìm thấy ảnh nào!")
            return
        
        # Xử lý từng ảnh với progress bar
        for img_name in tqdm(image_files, desc="Xử lý ảnh", unit="ảnh"):
            self._process_image(img_name)
        
        self.stats["without_faces"] = len(self.no_face_images)
        
        # Lưu logs
        self._save_logs()
        
        # In kết quả
        self._print_summary()
    
    def _save_logs(self):
        """Lưu các file log"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Log ảnh không có khuôn mặt
        if self.no_face_images:
            no_face_path = os.path.join(
                self.config.log_dir, 
                f"no_face_images_{timestamp}.txt"
            )
            with open(no_face_path, "w", encoding="utf-8") as f:
                f.write(f"# Ảnh không phát hiện khuôn mặt ({len(self.no_face_images)})\n")
                f.write(f"# Thời gian: {datetime.now()}\n\n")
                for name in self.no_face_images:
                    f.write(f"{name}\n")
        
        # Log ảnh lỗi
        if self.error_images:
            error_path = os.path.join(
                self.config.log_dir, 
                f"error_images_{timestamp}.txt"
            )
            with open(error_path, "w", encoding="utf-8") as f:
                f.write(f"# Ảnh gặp lỗi ({len(self.error_images)})\n")
                f.write(f"# Thời gian: {datetime.now()}\n\n")
                for name, error in self.error_images:
                    f.write(f"{name}: {error}\n")
        
        # Log thống kê tổng hợp
        stats_path = os.path.join(self.config.log_dir, f"statistics_{timestamp}.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
    
    def _print_summary(self):
        """In tóm tắt kết quả"""
        print("\n" + "=" * 60)
        print("✅ HOÀN THÀNH TẠO NHÃN")
        print("=" * 60)
        print(f"📊 Tổng số ảnh:              {self.stats['total']}")
        print(f"✅ Ảnh có khuôn mặt:         {self.stats['with_faces']} "
              f"({self.stats['with_faces']/max(self.stats['total'],1)*100:.1f}%)")
        print(f"❌ Ảnh không có khuôn mặt:   {self.stats['without_faces']} "
              f"({self.stats['without_faces']/max(self.stats['total'],1)*100:.1f}%)")
        print(f"⚠️  Ảnh lỗi:                  {self.stats['errors']}")
        print(f"👤 Tổng số khuôn mặt:        {self.stats['total_faces']}")
        
        if self.stats['with_faces'] > 0:
            avg_faces = self.stats['total_faces'] / self.stats['with_faces']
            print(f"📈 Trung bình khuôn mặt/ảnh: {avg_faces:.2f}")
        
        print(f"\n📁 Nhãn được lưu tại:        {self.config.lbl_dir}")
        print(f"📄 Logs được lưu tại:        {self.config.log_dir}")
        
        if self.config.save_visualization:
            print(f"🖼️  Visualizations tại:       {self.config.vis_dir}")
        
        print("=" * 60)


def main():
    """Hàm chính"""
    # Cấu hình
    config = Config(
        img_dir="face_detection_dataset/images",
        lbl_dir="face_detection_dataset/labels",
        log_dir="face_detection_dataset/logs",
        vis_dir="face_detection_dataset/visualizations",
        
        # Tham số detection
        scale_factor=1.1,
        min_neighbors=5,
        min_size=(30, 30),
        
        # Tùy chọn
        label_format="yolo",  # Hoặc "pascal_voc"
        save_visualization=False,  # Đặt True để xem kết quả
        create_empty_labels=True
    )
    
    # Chạy labeling
    labeler = FaceDetectionLabeler(config)
    labeler.process_all()


if __name__ == "__main__":
    main()