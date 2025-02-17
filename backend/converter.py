import os
from PIL import Image

def convert_images_to_jpeg(input_folder, output_folder, quality=95):
    """
    Converts all images in the input folder to JPEG format while preserving quality.

    :param input_folder: Path to the folder containing images.
    :param output_folder: Path to the folder where JPEG images will be saved.
    :param quality: Quality of the output JPEG images (default: 95).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    supported_formats = {"png", "jpg", "jpeg", "webp", "bmp", "tiff"}

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if os.path.isfile(file_path):
            try:
                with Image.open(file_path) as img:
                    img = img.convert("RGB")  # Convert to RGB to ensure JPEG compatibility
                    new_filename = os.path.splitext(filename)[0] + ".jpg"
                    output_path = os.path.join(output_folder, new_filename)
                    img.save(output_path, "JPEG", quality=quality)
                    print(f"Converted: {filename} -> {new_filename}")
            except Exception as e:
                print(f"Skipping {filename}: {e}")

if __name__ == "__main__":
    input_folder = "backend/images"  # Change this to your folder path
    output_folder = "backend/output_images"  # Change this to your desired output folder
    convert_images_to_jpeg(input_folder, output_folder)
