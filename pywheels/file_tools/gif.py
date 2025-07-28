# !! Under Construction
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PIL import Image
import time
import PIL


__all__ = [
    "gif_factory",
]


class GifCollector(FileSystemEventHandler):
    def __init__(self, src_path, output_dir):
        self.src_path = src_path
        self.output_dir = output_dir
        self.images = []
        self.index = 0
        self.verbose = True
        
    def collect(self):
        if os.path.exists(self.src_path):
            attempts = 114514  # 重试次数
            for _ in range(attempts):
                try:
                    # 尝试读取并保存图片
                    img = Image.open(self.src_path)
                    img.save(os.path.join(self.output_dir, f"{self.index}.png"))
                    self.images.append(os.path.join(self.output_dir, f"{self.index}.png"))
                    self.index += 1
                    break  # 成功后跳出循环
                except (OSError, PIL.UnidentifiedImageError) as e:
                    time.sleep(0.1) 
            

    def create_gif(self, gif_path, period):
        if not self.images:
            assert False, "Gif Factory has no images to create GIF!"

        if len(self.images) == 1:
            assert False, "Gif Factory has only one image to create GIF!"
        
        images = [Image.open(img) for img in self.images]
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=int((period * 1000) / len(self.images)), loop=0)
        if self.verbose: print(f"Gif Factory has collected {len(self.images)} images to create GIF.")
        if self.verbose: print(f"GIF saved at: {gif_path}")

        for img_path in self.images:
            if os.path.exists(img_path):
                os.remove(img_path) 
        self.images.clear()

class GifFactory:
    def __init__(self):
        self.verbose = True

    def start(self, monitor_path, storage_path):
        self.src_path = monitor_path
        self.output_dir = storage_path
        self.collector = GifCollector(monitor_path, storage_path)
        self.collector.verbose = self.verbose
        self.observer = Observer()
        self.observer.schedule(self.collector, os.path.dirname(self.src_path), recursive=False)
        self.observer.start()
        if self.verbose: print("Gif Factory starts monitoring...")
        
    def collect(self):
        self.collector.collect()

    def finish(self, output_path, period):
        self.observer.stop()
        self.observer.join()
        self.collector.create_gif(output_path, period)
        if self.verbose: print("Gif Factory stops monitoring and outputs a gif.")
        
gif_factory = GifFactory()