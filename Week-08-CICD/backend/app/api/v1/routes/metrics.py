from prometheus_client import Gauge, Histogram

gpu_allocated_metric = Gauge("process_vram_memory_GB", "GPU memory size in gigabytes.")

image_brightness_metric = Gauge("image_brightness", "Brightness of processed images")

brightness_histogram = Histogram("image_brightness_histogram", 
                                 "Histogram of image brightness",
                                 buckets=[100, 200, 255])