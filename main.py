from src.utils.helper.general import argParser

def main():
    args = argParser.make_parser().parse_args()
    print(args.video_path)

if __name__ == '__main__':
    main()