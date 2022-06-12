import argparse
import tqdm
import glob
import cv2

def make_parser():
    parser = argparse.ArgumentParser("convertImages2video tools")
    parser.add_argument(
        "-i",
        "--image_path",
        required=True,
        type=str,
        
        help="Input your image path to convert.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        required=True,
        type=str,
        
        help="Input your output path to save video.",
    )
    parser.add_argument(
        "-f",
        "--fps",
        type=int,
        default=15,
        help="Input your output path to save video.",
    )

    return parser

def convert_images_2_video(path: str, output_path:str, fps: int):
    """
    > It takes all the images in a folder and converts them into a video
    
    :param path: The path to the directory containing the images
    :type path: str
    :param output_path: The path to the output video file
    :type output_path: str
    """
    # List all image path
    image_file_paths = glob.glob(f'{path}/*.png') + glob.glob(f'{path}/*.jpg')
    image_file_paths.sort()

    # Check size of image
    img_array = []
    img = cv2.imread(image_file_paths[0])
    height, width, _ = img.shape
    size = (width,height)
    

    # Logging
    print(f'Number of image : {len(image_file_paths)}')
    outer = tqdm.tqdm(total=len(image_file_paths), desc='Image file', position=0)
    file_log = tqdm.tqdm(total=0, position=1, bar_format='{desc}')

    # Save image into array
    for image_file_path in image_file_paths:
        img = cv2.imread(image_file_path)
        img_array.append(img)
        file_log.set_description_str(f'Current image file: {image_file_path}')
        outer.update(1)
    
    # Logging
    outer = tqdm.tqdm(total=len(img_array), desc='Video Frame', position=0)
    video_log = tqdm.tqdm(total=0, position=1, bar_format='{desc}')

    # Save as video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_path,fourcc, fps, size)
    
    for frame_idx, img in enumerate(img_array):
        out.write(img)
        video_log.set_description_str(f'Current frame : {frame_idx+1}')
        outer.update(1)
    
    print(f'You can check it out on path : {output_path}')
    out.release()