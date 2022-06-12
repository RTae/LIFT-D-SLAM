import image_utils

def main():
    args = image_utils.make_parser().parse_args()
    image_utils.convert_images_2_video(args.image_path, args.output_path, args.fps)

if __name__ == '__main__':
    main()