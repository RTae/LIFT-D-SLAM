from src.utils.helper.general import argParser
import src.services as services

def main():
    args = argParser.make_parser().parse_args()

    c = services.Camera(args)
    while c.is_open():
        frame = c.get_frame()

if __name__ == '__main__':
    main()