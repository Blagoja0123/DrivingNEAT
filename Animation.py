import os
from PIL import Image
import glob


def create_gif_from_images(input_directory, output_filename="animation.gif", duration=500):

    image_extensions = ['*.png']
    image_files = []

    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_directory, extension)))
        image_files.extend(glob.glob(os.path.join(input_directory, extension.upper())))

    image_files.sort()

    if not image_files:
        print(f"No image files found in directory: {input_directory}")
        return

    print(f"Found {len(image_files)} images")

    images = []
    for image_file in image_files:
        try:
            img = Image.open(image_file)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(img)
            print(f"Loaded: {os.path.basename(image_file)}")
        except Exception as e:
            print(f"Error loading {image_file}: {e}")

    if not images:
        print("No valid images could be loaded")
        return

    print(f"\nCreating GIF with {len(images)} frames...")
    print(f"Duration per frame: {duration}ms")

    images[0].save(
        output_filename,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )

    print(f"GIF saved as: {output_filename}")

    total_time = len(images) * duration / 1000
    print(f"Total animation time: {total_time:.1f} seconds")


if __name__ == "__main__":
    input_dir = "visualization-snapshots"
    output_gif = "slow_animation.gif"
    frame_duration = 250


    create_gif_from_images(input_dir, output_gif, frame_duration)